#include "ProgramLink.h"

#include <spdlog/spdlog.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <sstream>
#include <vector>

namespace
{
long long now_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
}  // namespace

std::atomic<ProgramLink *> ProgramLink::active{nullptr};

ProgramLink::ProgramLink(const Config & cfg)
: cfg_(cfg)
{
}

ProgramLink::~ProgramLink()
{
    stop();
}

void ProgramLink::start()
{
    if (running_.load())
    {
        return;
    }

    rx_fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (rx_fd_ < 0)
    {
        spdlog::error("ProgramLink: socket() failed: {}", std::strerror(errno));
        return;
    }

    int reuse = 1;
    ::setsockopt(rx_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    // The receive blocks, so it needs a timeout to notice stop(). 100 ms costs nothing: this
    // thread does not drive anything, it only parks values for the policy thread to read.
    timeval tv{};
    tv.tv_sec = 0;
    tv.tv_usec = 100 * 1000;
    ::setsockopt(rx_fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<uint16_t>(cfg_.port));
    if (::bind(rx_fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0)
    {
        spdlog::error(
            "ProgramLink: cannot bind UDP {}: {} -- another controller or conductor is already "
            "using it. Network commands are off; the keyboard still drives.",
            cfg_.port, std::strerror(errno));
        ::close(rx_fd_);
        rx_fd_ = -1;
        return;
    }

    tx_fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (tx_fd_ < 0)
    {
        spdlog::warn("ProgramLink: no send socket ({}); state will not be published.",
                     std::strerror(errno));
    }

    reset();
    running_.store(true);
    thread_ = std::thread([this] { receive_loop(); });

    spdlog::info("ProgramLink: listening on UDP {}, state -> {}:{} (target valid for {:.2f}s)",
                 cfg_.port, cfg_.state_host, cfg_.state_port, cfg_.timeout_s);
}

void ProgramLink::stop()
{
    if (!running_.exchange(false))
    {
        return;
    }
    if (thread_.joinable())
    {
        thread_.join();
    }
    if (rx_fd_ >= 0)
    {
        ::close(rx_fd_);
        rx_fd_ = -1;
    }
    if (tx_fd_ >= 0)
    {
        ::close(tx_fd_);
        tx_fd_ = -1;
    }
    spdlog::info("ProgramLink: stopped after {} datagrams", datagrams_.load());
}

void ProgramLink::reset()
{
    vx_.store(0.0f);
    vy_.store(0.0f);
    wz_.store(0.0f);
    last_rx_ns_.store(-1);
    manual_stop_.store(false);
    std::lock_guard<std::mutex> lock(queue_mutex_);
    flips_.clear();
    stances_.clear();
}

void ProgramLink::receive_loop()
{
    std::vector<char> buffer(2048);
    while (running_.load())
    {
        const ssize_t n = ::recv(rx_fd_, buffer.data(), buffer.size() - 1, 0);
        if (n <= 0)
        {
            continue;  // timeout, or a stray error we cannot do anything about
        }
        datagrams_.fetch_add(1);
        buffer[static_cast<size_t>(n)] = '\0';

        std::istringstream stream(std::string(buffer.data(), static_cast<size_t>(n)));
        std::string line;
        while (std::getline(stream, line))
        {
            if (!line.empty() && line.back() == '\r')
            {
                line.pop_back();
            }
            if (!line.empty())
            {
                handle_line(line);
            }
        }
    }
}

void ProgramLink::handle_line(const std::string & line)
{
    std::istringstream parts(line);
    std::string verb;
    parts >> verb;

    if (verb == "VEL")
    {
        float vx = 0.0f, vy = 0.0f, wz = 0.0f;
        if (!(parts >> vx >> vy >> wz))
        {
            spdlog::warn("ProgramLink: malformed '{}'", line);
            return;
        }
        vx_.store(vx);
        vy_.store(vy);
        wz_.store(wz);
        last_rx_ns_.store(now_ns());
    }
    else if (verb == "STOP")
    {
        vx_.store(0.0f);
        vy_.store(0.0f);
        wz_.store(0.0f);
        last_rx_ns_.store(now_ns());
        std::lock_guard<std::mutex> lock(queue_mutex_);
        flips_.clear();
        stances_.clear();
    }
    else if (verb == "FLIP")
    {
        std::string kind;
        if (!(parts >> kind))
        {
            spdlog::warn("ProgramLink: FLIP with no kind");
            return;
        }
        std::lock_guard<std::mutex> lock(queue_mutex_);
        // Bounded so a conductor stuck in a loop cannot build a backlog the robot would then work
        // through after the operator has stopped it.
        if (flips_.size() < 4)
        {
            flips_.push_back(kind);
        }
    }
    else if (verb == "STANCE")
    {
        std::string which;
        if (!(parts >> which))
        {
            spdlog::warn("ProgramLink: STANCE with no side");
            return;
        }
        float sign = 0.0f;
        if (which == "front")      sign = 1.0f;
        else if (which == "hind")  sign = -1.0f;
        else if (which == "off")   sign = 0.0f;
        else
        {
            spdlog::warn("ProgramLink: STANCE '{}' is not front/hind/off", which);
            return;
        }
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (stances_.size() < 4)
        {
            stances_.push_back(sign);
        }
    }
    else if (verb == "RESUME")
    {
        if (manual_stop_.exchange(false))
        {
            spdlog::info("ProgramLink: manual stop cleared, network commands live again");
        }
    }
    else if (verb == "PING")
    {
        // Liveness only; the state stream is the reply.
    }
    else
    {
        spdlog::warn("ProgramLink: unknown command '{}'", verb);
    }
}

bool ProgramLink::velocity(std::array<float, 3> & out) const
{
    if (manual_stop_.load())
    {
        return false;
    }
    const long long stamp = last_rx_ns_.load();
    if (stamp < 0)
    {
        return false;
    }
    const double age_s = static_cast<double>(now_ns() - stamp) * 1e-9;
    if (age_s > cfg_.timeout_s)
    {
        return false;
    }
    out = {vx_.load(), vy_.load(), wz_.load()};
    return true;
}

bool ProgramLink::take_flip(std::string & kind)
{
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (flips_.empty())
    {
        return false;
    }
    kind = flips_.front();
    flips_.pop_front();
    return true;
}

bool ProgramLink::take_stance(float & sign)
{
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (stances_.empty())
    {
        return false;
    }
    sign = stances_.front();
    stances_.pop_front();
    return true;
}

void ProgramLink::note_manual_stop()
{
    if (!manual_stop_.exchange(true))
    {
        spdlog::warn("ProgramLink: [Space] pressed -- network commands held off until RESUME");
        std::lock_guard<std::mutex> lock(queue_mutex_);
        flips_.clear();
        stances_.clear();
    }
}

void ProgramLink::publish(const std::string & line)
{
    if (tx_fd_ < 0)
    {
        return;
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(cfg_.state_port));
    if (::inet_pton(AF_INET, cfg_.state_host.c_str(), &addr.sin_addr) != 1)
    {
        return;
    }
    ::sendto(tx_fd_, line.data(), line.size(), MSG_DONTWAIT,
             reinterpret_cast<sockaddr *>(&addr), sizeof(addr));
}
