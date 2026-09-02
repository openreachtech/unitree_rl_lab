#pragma once

#include <string>
#include <vector>
#include <deque>
#include <unordered_set>
#include <mutex>
#include <atomic>
#include <termios.h>
#include <unistd.h>
#include <thread>


/**
 * @brief Maintain a keyboard reading thread.
 * And get the latest key value.
 */
class Keyboard
{
public:
  Keyboard()
  {
    tcgetattr( fileno( stdin ), &_oldSettings );
    _newSettings = _oldSettings;
    _oldSettings.c_lflag |= ( ICANON |  ECHO);
    _newSettings.c_lflag &= (~ICANON & ~ECHO);

    _startKey();

    _thread_running  = true;
    _readThread = std::thread([this] {
      while (_running) {
        _read();
      }
    });
  }

  ~Keyboard()
  {
    _thread_running = false;
    _pauseKey();
  }

  void update()
  {
    if(_key != _last_key)
    {
      on_pressed = _key != "";
      on_released = _key == "";
    }
    else
    {
      on_pressed = false;
      on_released = false;
    }
    
    _last_key = _key;
  }

  /**
   * @brief Get the current key value
   * 
   * @return std::string 
   */
  std::string key() const { return _key; };

  /** @brief Latched keys (insert on press; cleared on space). Thread-safe. */
  bool pressed(const std::string& key) const
  {
    std::lock_guard<std::mutex> lock(_pressed_mutex);
    if (key == "space")
    {
      return _pressed_keys.count(" ") > 0;
    }
    return _pressed_keys.count(key) > 0;
  }

  void clear_pressed_keys()
  {
    std::lock_guard<std::mutex> lock(_pressed_mutex);
    _pressed_keys.clear();
  }

  /** @brief One-shot version of pressed(): true (once) if key is latched, and removes it
   *  from the latched set so holding the key down has no repeated effect. Use this instead
   *  of pressed() for keys that select between mutually-exclusive discrete states (e.g. a
   *  height preset), where -- unlike f/b/l/r/y/u's additive velocity components -- letting
   *  more than one stay latched at once would be ambiguous rather than combine sensibly. */
  bool consume(const std::string& key)
  {
    std::lock_guard<std::mutex> lock(_pressed_mutex);
    const std::string& k = (key == "space") ? " " : key;
    return _pressed_keys.erase(k) > 0;
  }

  /** @brief True once after space bar; clears latched motion keys. */
  bool consume_velocity_stop()
  {
    return _velocity_stop_requested.exchange(false);
  }

  /**
   * @brief Get the String object from keyboard 
   * 
   * @param slogan Used to prompt the user for input
   * @return std::string 
   */
  std::string getString(std::string slogan)
  {
    // Stop reading keyboard value
    _running = false;
    _pauseKey();

    std::string stringtemp;
    std::cout << slogan << std::endl;// prompt
    std::getline(std::cin, stringtemp);

    // Restart reading keyboard value
    _startKey();
    _running = true;

    return stringtemp;
  }

  /**
   * flags; available after update()
   */
  bool on_pressed = false;
  bool on_released = false;

  private:
  bool _thread_running = false;
  bool _running = false;
  std::thread _readThread;

  void _read()
  {
    if(_running)
    {
      FD_ZERO(&_fd_set);
      FD_SET( fileno(stdin), &_fd_set);

      _tv.tv_sec = 0;
      _tv.tv_usec = 80000;

      if(select(fileno(stdin)+1, &_fd_set, NULL, NULL, &_tv))
      {
        // Read the key value into _c
        int res = read( fileno(stdin), &_c, 1 );

        // Parser the key value
        if(_c != '\033') {
          // This is a normal key
          _key = std::string(1, _c);
          _register_key_event(_key);
        }else{
          // This is a special key
          int m = read(fileno(stdin), &_c, 1);
          if(_c == '[')
          {
            m = read(fileno(stdin), &_c, 1);
            switch (_c)
            {
            case 'A': _key = "up";    break;
            case 'B': _key = "down";  break;
            case 'C': _key = "right"; break;
            case 'D': _key = "left";  break;
            default:  _key = "";      break;
            }
            if (!_key.empty())
            {
              _register_key_event(_key);
            }
          }
        }
      }else{
        _key = "";
      }
      // std::cout << "key: "<< key() << std::endl;
    }
  }

  /**
   * @brief Restore keyboard default settings.
   */
  void _pauseKey()
  {
    tcsetattr( fileno( stdin ), TCSANOW, &_oldSettings );
    _running = false;
  }

  /**
   * @brief Disable canonical mode and echoing of input characters.
   */
  void _startKey()
  {
    tcsetattr( fileno( stdin ), TCSANOW, &_newSettings );
    _running = true;
  }

  void _register_key_event(const std::string& key)
  {
    std::lock_guard<std::mutex> lock(_pressed_mutex);
    if (key == " " || key == "space")
    {
      _pressed_keys.clear();
      _velocity_stop_requested = true;
      return;
    }
    if (!key.empty())
    {
      _pressed_keys.insert(key);
    }
  }

  fd_set _fd_set;
  char _c = '\0';
  std::string _key, _last_key;

  mutable std::mutex _pressed_mutex;
  std::unordered_set<std::string> _pressed_keys;
  std::atomic<bool> _velocity_stop_requested{false};

  termios _oldSettings, _newSettings;
  timeval _tv;
};