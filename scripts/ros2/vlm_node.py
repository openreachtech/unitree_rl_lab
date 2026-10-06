"""SigLIP 2 scorer: turns the front camera into a value map and an object fix.

One model, two answers, one forward pass. See doc/design/vlfm_nav.md 7 for the design;
what runs here is:

    image -> whole frame + 4 vertical strips -> SigLIP 2 (batch of 5)
      |-- strip scores  -> painted into four 22.5 deg sub-fans of the value map
      |-- whole score   -> floor under all four (scene cues are not localised)
      '-- max strip > threshold -> bearing -> raycast /map -> object pose

Runs in the env_isaaclab venv against Isaac's bundled ROS 2, but does NOT start Isaac --
see _bundled_ros.py. That means it also runs happily against a rosbag, which is the
point: prompts, thresholds and models are all tuning knobs, and tuning them needs many
passes over the same trajectory, not many trajectories.

    # live
    python scripts/ros2/vlm_node.py --target toilet
    # offline, against a recording (ros2 bag play --clock in another shell)
    python scripts/ros2/vlm_node.py --target toilet --use-sim-time

Publishes:
    /vlfm/value_map         nav_msgs/OccupancyGrid   0-100 where looked at, -1 elsewhere
    /vlfm/value_confidence  nav_msgs/OccupancyGrid   how hard it was looked at
    /vlfm/vlm/debug/image_raw    sensor_msgs/Image       strips, scores, detection marked
    /vlfm/vlm/debug/camera_info  sensor_msgs/CameraInfo  forwarded; Foxglove wants the pair
    /vlfm/detection         geometry_msgs/PoseStamped one threshold crossing (map frame).
                                                      NOT a goal -- see the publisher.
    /vlfm/vlm/status        std_msgs/String          one line per scored frame

Why two grids: a low value alone is ambiguous -- it can mean "looked at, nothing there"
or "never looked at". Confidence separates them, and it is also the only way to see
whether the occlusion cut is working, because a fan bleeding through a wall shows up
there before it shows up anywhere else.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import deque

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from _bundled_ros import ensure_bundled_ros  # noqa: E402

ensure_bundled_ros()  # may re-exec; keep everything expensive below this line

import numpy as np  # noqa: E402
import rclpy  # noqa: E402
import torch  # noqa: E402
from geometry_msgs.msg import PoseStamped  # noqa: E402
from nav_msgs.msg import OccupancyGrid  # noqa: E402
from PIL import Image as PILImage  # noqa: E402
from PIL import ImageDraw  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from tf2_ros import Buffer, TransformListener  # noqa: E402

# value_map.py is pure numpy, so the py3.10 colcon package imports cleanly here.
sys.path.insert(0, os.path.join(_HERE, "..", "..", "ros2", "vlfm_nav"))
from vlfm_nav.value_map import ValueMap  # noqa: E402


def embed(out) -> torch.Tensor:
    """Pull the embedding out of get_*_features, whichever transformers this is.

    4.x returned the tensor; 5.x returns the model output object and leaves the
    pooling to the caller. The two venvs here are not on the same major
    (env_isaaclab is on 5.x because marker-pdf needs it), so handle both rather than
    pin a version to a detail this small.
    """
    t = getattr(out, "pooler_output", out)
    return t / t.norm(dim=-1, keepdim=True)


def yaw_of(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))


class VlmNode(Node):
    def __init__(self, a):
        super().__init__("vlfm_vlm")
        self.a = a
        # rclpy declares use_sim_time itself, so set it rather than redeclaring. It has
        # to be on for bag replay: every TF lookup here keys off the image stamp, which
        # is sim time, and against the wall clock they are years apart.
        self.set_parameters([rclpy.parameter.Parameter(
            "use_sim_time", rclpy.Parameter.Type.BOOL, a.use_sim_time)])

        self.get_logger().info(f"loading {a.model} ...")
        from transformers import AutoModel, AutoProcessor

        self.proc = AutoProcessor.from_pretrained(a.model)
        self.model = AutoModel.from_pretrained(a.model, dtype=torch.float16).to(a.device).eval()
        # Text side is encoded once and cached: the per-frame cost is one image forward
        # plus a 5xD @ Dx2 matmul, so carrying a second prompt is free.
        texts = [a.detect_prompt.format(a.target), a.value_prompt.format(a.target)]
        ti = self.proc(text=texts, padding="max_length", max_length=64, return_tensors="pt")
        ti = {k: v.to(a.device) for k, v in ti.items()}
        with torch.no_grad():
            self.temb = embed(self.model.get_text_features(**ti)).detach().float()
        # logit_scale/bias are nn.Parameters; without detach every product carries grad.
        self.scale = self.model.logit_scale.exp().detach().float()
        self.bias = self.model.logit_bias.detach().float()
        self.get_logger().info(f"target='{a.target}'  detect={texts[0]!r}  value={texts[1]!r}")

        self.vmap = ValueMap(
            size_m=a.map_size, resolution=a.map_res,
            origin=(-a.map_size / 2, -a.map_size / 2),
            hfov_rad=math.radians(a.hfov_deg), n_strips=a.strips, max_range_m=a.value_range,
        )
        self.occ = None  # (array, origin, resolution) from the latest /map
        self.last_pose = None  # (x, y, yaw) of the last scored frame
        self.last_m2o = None  # map->odom, watched for loop-closure jumps
        self.last_t = -1e9
        self.pending = deque(maxlen=30)  # frames waiting to ripen; see on_image
        self.last_verdict = (np.zeros(a.strips), 0.0, None)  # strips, whole, hit
        self.n_scored = 0

        self.tf = Buffer(cache_time=rclpy.duration.Duration(seconds=30.0))
        # spin_thread: the listener gets its own executor. Without it, TF is processed
        # on the same single thread as the image callback -- which both blocks for tens
        # of ms of inference and then blocks again waiting for the very transform it is
        # starving. Measured against a bag: TF fell 17.7 s behind the images and every
        # lookup failed as "extrapolation into the future", a stall that feeds itself.
        TransformListener(self.tf, self, spin_thread=True)

        img_qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)
        map_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Image, a.image_topic, self.on_image, img_qos)
        # Forwarded so the debug image has a CameraInfo beside it under the same
        # namespace. See the publisher below for why that matters.
        self.caminfo = None
        self.create_subscription(
            CameraInfo, a.image_topic.rsplit("/", 1)[0] + "/camera_info",
            lambda m: setattr(self, "caminfo", m), img_qos)
        self.create_subscription(OccupancyGrid, "/map", self.on_map, map_qos)

        self.pub_value = self.create_publisher(OccupancyGrid, "/vlfm/value_map", 1)
        self.pub_conf = self.create_publisher(OccupancyGrid, "/vlfm/value_confidence", 1)
        # Named <ns>/image_raw with a <ns>/camera_info beside it, not /vlfm/vlm/debug_image.
        # Foxglove's Image panel offers the one shape and not the other -- established by
        # straight A/B on 2026-10-02/05 with the message bytes identical either way
        # (768x384 rgb8, step 2304, same frame), the topic advertised, and data flowing:
        # renamed to this it appears, renamed back it does not. Do not "tidy" this back
        # to a flat name.
        self.pub_debug = self.create_publisher(Image, "/vlfm/vlm/debug/image_raw", img_qos)
        self.pub_debug_info = self.create_publisher(
            CameraInfo, "/vlfm/vlm/debug/camera_info", img_qos)
        # RAW detections, one per frame that crosses the threshold -- NOT the thing the
        # robot drives to. The explorer only commits after several of these agree in
        # time and space (target_confirm_n / _window_s / _cluster_m), and publishes the
        # committed one on /vlfm/target. Keeping the two apart matters for more than
        # tidiness: a single crossing used to show up in Foxglove as an arrow that
        # looked exactly like a goal, so every unconfirmed glimpse read as "the robot
        # found it and is ignoring it".
        #
        # Latched: a detection is a single message fired the instant it happens, and a
        # panel opened a minute later should still see where the thing was. Without
        # transient-local the topic looks permanently empty to anyone who was not already
        # watching at exactly the right moment.
        self.pub_target = self.create_publisher(
            PoseStamped, "/vlfm/detection",
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.pub_status = self.create_publisher(String, "/vlfm/vlm/status", 10)
        self.create_timer(1.0 / a.tick_hz, self.tick)
        self.create_timer(1.0 / a.publish_hz, self.publish_maps)
        self.get_logger().info("ready")

    # ------------------------------------------------------------------- inputs
    def on_map(self, m: OccupancyGrid):
        self.occ = (
            np.asarray(m.data, dtype=np.int16).reshape(m.info.height, m.info.width),
            (m.info.origin.position.x, m.info.origin.position.y),
            m.info.resolution,
        )

    def _follow_loop_closure(self) -> None:
        """Keep the painted map aligned when SLAM corrects map->odom.

        Everything here is painted in the map frame, so a correction silently displaces
        all of it -- the mechanism that once painted 52 m^2 of free floor lethal in the
        global costmap. The first version discarded the map on every jump, which turned
        out to be worse than the disease: over the 2026-10-01 run slam_toolbox crossed
        the threshold four times in 25 minutes, so 36,758 cell-writes left 107 cells
        standing. The correction is known, so move the paint with it; only a jump too
        large for a rigid approximation to mean anything still throws it away.
        """
        try:
            t = self.tf.lookup_transform("map", "odom", rclpy.time.Time()).transform
        except Exception:
            return
        cur = (t.translation.x, t.translation.y, yaw_of(t.rotation))
        prev, self.last_m2o = self.last_m2o, cur
        if prev is None:
            return
        d = math.hypot(cur[0] - prev[0], cur[1] - prev[1])
        dth = (cur[2] - prev[2] + math.pi) % (2 * math.pi) - math.pi
        if d <= self.a.closure_trans and abs(dth) <= self.a.closure_rot:
            return
        if d > self.a.closure_discard:
            self.vmap.clear()
            self.get_logger().warn(f"map->odom jumped {d:.2f} m; value map discarded")
            return
        # C = T_new * T_old^-1, the map-frame correction the closure applied.
        c, s_ = math.cos(dth), math.sin(dth)
        cx = cur[0] - (c * prev[0] - s_ * prev[1])
        cy = cur[1] - (s_ * prev[0] + c * prev[1])
        self.vmap.transform(cx, cy, dth)
        self.get_logger().info(
            f"map->odom moved {d:.2f} m / {math.degrees(dth):.1f} deg; value map followed")

    # -------------------------------------------------------------------- score
    def on_image(self, msg: Image):
        """Stash only. Everything expensive happens on the timer below.

        An image arrives before the transform that explains it -- measured over the
        2026-10-01 bag, odom->base is stamped exactly 0.1 s behind every single frame,
        which is RKO-LIO's pipeline latency, not a backlog. Blocking here to wait for
        it is what killed the first two attempts: the wait ran on the executor that also
        feeds the TF buffer, so waiting guaranteed the thing being waited for could not
        arrive, and the lag grew to 6 s and stayed there.

        So: never block in a callback. Queue the frame, let it ripen, and process it
        from a timer once its transform is certainly in.

        A queue and not a single slot. With one slot the frame is replaced by the next
        arrival before it can ripen -- images are 0.2 s apart and the wait is 0.25 s, so
        the held frame's age went 0.1, 0.2, then reset, forever. Every lookup failed by
        exactly the 0.1 s of LIO lag and the value map stayed empty while the robot
        walked past the thing it was looking for.
        """
        self.pending.append(msg)

    def tick(self):
        """Take the newest ripe frame; drop whatever is older than it, unlooked at.

        Dropping is deliberate: a backlog means the scorer is behind, and the freshest
        observation is always the one worth having. The motion gate throws most frames
        away in any case.
        """
        if self.occ is None or not self.pending:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        msg = None
        while self.pending:
            head = self.pending[0]
            age = now - (head.header.stamp.sec + head.header.stamp.nanosec * 1e-9)
            if age < self.a.tf_delay:
                break                     # this and everything after it is still green
            msg = self.pending.popleft()
        if msg is None:
            return
        img = PILImage.frombytes("RGB", (msg.width, msg.height), bytes(msg.data))
        # Draw first, with whatever the last verdict was. Scoring is rate- and
        # motion-gated down to a frame every several seconds, and a debug view that
        # updates that rarely is not a view -- measured 0.12 Hz on a live run, with gaps
        # of half a minute. The picture is cheap; the inference behind it is not, so the
        # picture tracks the camera and the numbers age visibly between scores.
        self._publish_debug(msg, img, *self.last_verdict)

        # Rate gate, unconditionally, so a run of failures cannot turn into a busy loop
        # that starves every other callback.
        if now - self.last_t < 1.0 / self.a.max_hz:
            return
        self.last_t = now
        # TF at the IMAGE's stamp, never "now" -- the pose that took the picture is the
        # pose the paint belongs to, and over a moving base they differ.
        stamp = rclpy.time.Time.from_msg(msg.header.stamp)
        try:
            tr = self.tf.lookup_transform("map", msg.header.frame_id, stamp).transform
        except Exception as e:
            self.get_logger().warn(f"no TF map<-{msg.header.frame_id}: {e}", throttle_duration_sec=5.0)
            return
        # The image frame is the optical one (+z forward); our bearing maths wants the
        # body-style `camera` frame (+x forward), so take the yaw from that instead.
        try:
            body = self.tf.lookup_transform("map", "camera", stamp).transform
        except Exception:
            body = tr
        x, y, yaw = tr.translation.x, tr.translation.y, yaw_of(body.rotation)

        # Motion gate: standing still re-scores an identical picture, which adds no
        # information but does keep raising the confidence of whatever is in front.
        if self.last_pose is not None:
            dx, dy, dyaw = x - self.last_pose[0], y - self.last_pose[1], yaw - self.last_pose[2]
            dyaw = abs((dyaw + math.pi) % (2 * math.pi) - math.pi)
            if math.hypot(dx, dy) < self.a.move_gate and dyaw < self.a.turn_gate:
                return
        self.last_pose = (x, y, yaw)

        self._follow_loop_closure()

        W, H = img.size
        n = self.a.strips
        strips = [img.crop((i * W // n, 0, (i + 1) * W // n, H)) for i in range(n)]
        ii = self.proc(images=[img] + strips, return_tensors="pt")
        ii = {k: (v.to(self.a.device, torch.float16) if v.is_floating_point() else v.to(self.a.device))
              for k, v in ii.items()}
        with torch.no_grad():
            ie = embed(self.model.get_image_features(**ii)).float()
            p = torch.sigmoid(ie @ self.temb.T * self.scale + self.bias).cpu().numpy()
        detect, value = p[:, 0], p[:, 1]
        strip_detect, whole_value = detect[1:], float(value[0])

        # Strips carry the angular resolution; the whole frame is a floor under all of
        # them. Measured 2026-10-01: strips find objects (bed 0.046 whole -> 0.461 strip)
        # but miss room-level cues entirely (a kitchen scores 0.22 at best on a strip,
        # because a quarter-frame crop of a counter is not "a photo of a kitchen"), and
        # room-level cues are exactly what a value map wants.
        painted = self.vmap.paint(
            np.array([x, y]), yaw, np.maximum(strip_detect, whole_value), *self.occ
        )
        self.n_scored += 1

        hit = self._detect(strip_detect, np.array([x, y]), yaw, msg.header.stamp)
        self.last_verdict = (strip_detect, whole_value, hit)
        self._publish_debug(msg, img, strip_detect, whole_value, hit)
        s = " ".join(f"{v:.3f}" for v in strip_detect)
        self.pub_status.publish(String(data=(
            f"#{self.n_scored} pose=({x:.2f},{y:.2f},{math.degrees(yaw):.0f}deg) "
            f"strips=[{s}] whole={whole_value:.3f} cells={painted}"
            f"{' HIT' if hit else ''}")))

    # ------------------------------------------------------------------- detect
    def _detect(self, strip_scores, cam_xy, yaw, stamp):
        """Score-weighted bearing of the target, then the first wall along it.

        The centroid rather than the argmax: with the strips 22.5 deg wide, weighting
        the two neighbours resolves the bearing far finer than the strip count.

        No depth camera is involved. The range comes from the same occupancy raycast the
        value fan uses, so it costs nothing extra -- and it lands in the map frame
        already, skipping a whole class of camera-to-map calibration bugs. The caveat is
        that it returns the FIRST obstacle along the bearing, which is the target only
        when the target is what we are looking at; that is true by construction here,
        since we only run when the score is high.
        """
        best = float(strip_scores.max())
        if best < self.a.detect_threshold:
            return None
        n = len(strip_scores)
        w = np.where(strip_scores >= self.a.detect_threshold, strip_scores, 0.0)
        centres = self.a.hfov_deg / 2 - (np.arange(n) + 0.5) * (self.a.hfov_deg / n)
        rel = math.radians(float((w * centres).sum() / w.sum()))
        theta = yaw + rel

        occ, org, res = self.occ
        steps = np.arange(1, int(self.a.detect_range / res) + 1) * res
        px, py = cam_xy[0] + np.cos(theta) * steps, cam_xy[1] + np.sin(theta) * steps
        ci, cj = np.floor((px - org[0]) / res).astype(int), np.floor((py - org[1]) / res).astype(int)
        h, wd = occ.shape
        ok = (ci >= 0) & (ci < wd) & (cj >= 0) & (cj < h)
        if not ok.any():
            return None
        vals = np.full(len(steps), -1, dtype=np.int16)
        vals[ok] = occ[cj[ok], ci[ok]]
        solid = np.where(vals >= 50)[0]
        if len(solid) == 0:
            return None
        r = steps[solid[0]]

        m = PoseStamped()
        m.header.stamp = stamp
        m.header.frame_id = "map"
        m.pose.position.x = float(cam_xy[0] + math.cos(theta) * r)
        m.pose.position.y = float(cam_xy[1] + math.sin(theta) * r)
        m.pose.orientation.z, m.pose.orientation.w = math.sin(theta / 2), math.cos(theta / 2)
        self.pub_target.publish(m)
        return (m.pose.position.x, m.pose.position.y, best)

    # ------------------------------------------------------------------ outputs
    def _publish_debug(self, msg, img, strip_scores, whole, hit):
        """The frame with the strip grid, each strip's score, and the verdict burned in.

        This is the artifact that answers "is it seeing what I think it is seeing"
        without reading a single number off a plot.

        Published unconditionally, even though drawing for nobody is waste. Gating it on
        the subscriber count deadlocks the thing it was meant to serve: Foxglove's Image
        panel will not offer a topic that has never carried a message, so the topic could
        never be selected, so it never got a subscriber, so it never published. At <=2 Hz
        the cost is small and being able to look at it is the entire point.

        (The camera's own image_raw keeps its gate -- the VLM node subscribes to that in
        code, not by someone picking it out of a dropdown, so the same trap does not
        apply.)
        """
        d = img.copy()
        dr = ImageDraw.Draw(d)
        W, H = d.size
        n = len(strip_scores)
        best = int(np.argmax(strip_scores))
        for i, s in enumerate(strip_scores):
            x0, x1 = i * W // n, (i + 1) * W // n
            on = s >= self.a.detect_threshold
            dr.rectangle([x0, 0, x1 - 1, H - 1],
                         outline=(255, 60, 40) if on else (90, 90, 90),
                         width=4 if i == best and on else 1)
            dr.text((x0 + 6, 6), f"{s:.3f}", fill=(255, 60, 40) if on else (230, 230, 230))
        dr.text((6, H - 28), f"{self.a.target}  whole={whole:.3f}" + (
            f"  HIT -> ({hit[0]:.2f}, {hit[1]:.2f})" if hit else ""), fill=(255, 255, 0))
        out = Image(header=msg.header, height=H, width=W, encoding="rgb8",
                    is_bigendian=0, step=W * 3)
        out.data = np.asarray(d, dtype=np.uint8).tobytes()
        self.pub_debug.publish(out)
        if self.caminfo is not None:
            self.caminfo.header = msg.header
            self.pub_debug_info.publish(self.caminfo)

    def publish_maps(self):
        if self.n_scored == 0:
            return
        val, cnf = self.vmap.as_occupancy()
        for pub, data in ((self.pub_value, val), (self.pub_conf, cnf)):
            g = OccupancyGrid()
            g.header.stamp = self.get_clock().now().to_msg()
            g.header.frame_id = "map"
            g.info.resolution = self.vmap.res
            g.info.width = g.info.height = self.vmap.n
            g.info.origin.position.x = float(self.vmap.origin[0])
            g.info.origin.position.y = float(self.vmap.origin[1])
            g.info.origin.orientation.w = 1.0
            g.data = data.ravel().tolist()
            pub.publish(g)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", default="toilet", help="what to look for, in English")
    p.add_argument("--model", default="google/siglip2-so400m-patch16-naflex",
                   help="NaFlex keeps the strips' native aspect ratio; the fixed-384"
                        " checkpoints squash a 1:2 strip into a square.")
    p.add_argument("--detect-prompt", default="This is a photo of a {}.",
                   help="Measured best of six on 2026-09-30: 0.390 on a strip showing the"
                        " object vs 0.0035 on one that does not. VLFM's own"
                        " 'Seems like there is a {} ahead.' scores 40x lower here -- it was"
                        " written for BLIP-2 ITM, not SigLIP.")
    p.add_argument("--value-prompt", default="This is a photo of a {}.")
    p.add_argument("--detect-threshold", type=float, default=0.1,
                   help="Validated over 2,465 frames on 2026-10-01: the control target"
                        " ('elephant', absent from the apartment) never crosses it, while"
                        " every real object is picked up in proportion to how often it"
                        " was actually in view.")
    p.add_argument("--image-topic", default="/vlfm/camera/image_raw")
    p.add_argument("--hfov-deg", type=float, default=90.0)
    p.add_argument("--strips", type=int, default=4)
    p.add_argument("--move-gate", type=float, default=0.5, help="metres before re-scoring")
    p.add_argument("--turn-gate", type=float, default=0.35, help="radians before re-scoring")
    p.add_argument("--max-hz", type=float, default=1.0,
                   help="ceiling on scoring rate. Not the real limiter: the motion gate"
                        " below throws most frames away first, and the measured rate on a"
                        " live run is 0.12 Hz. This only caps the burst when the robot is"
                        " turning fast enough to clear the gate every frame.")
    p.add_argument("--value-range", type=float, default=5.0, help="how far the value fan reaches")
    p.add_argument("--detect-range", type=float, default=8.0, help="how far to raycast for the object")
    p.add_argument("--map-size", type=float, default=60.0)
    p.add_argument("--map-res", type=float, default=0.2)
    p.add_argument("--publish-hz", type=float, default=1.0)
    p.add_argument("--closure-trans", type=float, default=0.5,
                   help="map->odom jump that discards the value map")
    p.add_argument("--closure-rot", type=float, default=0.3)
    p.add_argument("--closure-discard", type=float, default=3.0,
                   help="a map->odom jump bigger than this is not a correction worth"
                        " following rigidly; throw the paint away instead")
    p.add_argument("--tf-delay", type=float, default=0.25,
                   help="how long to let a frame ripen before scoring it, so the"
                        " transform for its stamp has certainly arrived. RKO-LIO runs a"
                        " constant 0.1 s behind (measured over 2,465 frames), so this is"
                        " a fixed offset, not a guess at a backlog.")
    p.add_argument("--tick-hz", type=float, default=10.0,
                   help="how often to check whether the held frame is ready")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--use-sim-time", action="store_true",
                   help="required when replaying a bag (play it with --clock)")
    a = p.parse_args()

    rclpy.init()
    node = VlmNode(a)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # A SIGTERM during spin can shut the context down before we get here, and the
        # second call then raises -- noise that reads like a crash in the log.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
