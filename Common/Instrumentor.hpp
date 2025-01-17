/**
 * @file Instrumentor.hpp
 *
 * @brief Basic Instrumentation Profiler for C++ | Macro-fied
 *
 * Upon execution of profile macro embedded source code, all the profiling data
 * is dumped to a JSON file compatible with Chrome Tracing tool. The JSON file can
 * then be loaded to chrome tracing (chrome://tracing) to visualise the data.
 *
 * @author TheCherno       (primary author)
 * @author Ashwin A Nayar  (github.com/nocoinman)
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <fstream>
#include <string>
#include <thread>
#include <map>

#define ALIAS_TEMPLATE_FUNC(highLevelF, lowLevelF)                                                                     \
  template <typename... Args>                                                                                          \
  inline auto highLevelF(Args &&... args)->decltype(lowLevelF(std::forward<Args>(args)...))                            \
  {                                                                                                                    \
    return lowLevelF(std::forward<Args>(args)...);                                                                     \
  }

#define CONCAT(a, b) CONCAT_INNER(a, b)
#define CONCAT_INNER(a, b) a##b

///////////////////////////////////////////////////////////////////////////////////////////////////

struct ProfileResult
{
  std::string Name;
  long long   Start, End;
  uint32_t    ThreadID;
};

struct InstrumentationSession
{
  std::string Name;
  std::string Filename; // filepath without extension
};

template <typename A, typename B>
std::pair<B, A>
flipPair(const std::pair<A, B> & p)
{
  return std::pair<B, A>(p.second, p.first);
}

template <typename A, typename B>
std::multimap<B, A, std::greater<long long>>
flipMap(const std::map<A, B> & src)
{
  std::multimap<B, A, std::greater<long long>> dst;
  std::transform(src.begin(), src.end(), std::inserter(dst, dst.begin()), flipPair<A, B>);
  return dst;
}

class Instrumentor
{
public:
  Instrumentor(const Instrumentor &) = delete;

  static void
  beginSession(const std::string & name, const std::string & filename)
  {
    get().beginSessionImpl(name, filename);
  }

  static void
  endSession()
  {
    get().endSessionImpl();
  }

  static void
  write(const ProfileResult & r)
  {
    get().writeProfile(r);
  }

private:
  Instrumentor()
    : m_CurrentSession(nullptr)
    , m_ProfileCount(0)
  {}

  static Instrumentor &
  get()
  {
    static Instrumentor instance;
    return instance;
  }

  void
  beginSessionImpl(const std::string & name, const std::string & filename)
  {
    if (!m_SessionActive)
    {
      m_SessionActive = true;
      m_OutputStream.open(filename + ".json");
      writeHeader();
      m_CurrentSession = new InstrumentationSession{ name, filename };
    }
  }

  void
  endSessionImpl()
  {
    if (m_SessionActive)
    {
      writeFooter();
      m_OutputStream.close();
      writeSummary();
      delete m_CurrentSession;
      m_CurrentSession = nullptr;
      m_ProfileCount = 0;
      m_TotalTime = 0L;
      m_SessionSummary = std::map<std::string, long long>{};
      m_SessionActive = false;
    }
  }

  void
  writeProfile(const ProfileResult & result)
  {
    if (!m_SessionActive)
      return;

    if (m_ProfileCount++ > 0)
      m_OutputStream << ",";

    std::string name = result.Name;
    std::replace(name.begin(), name.end(), '"', '\'');

    m_OutputStream << "{";
    m_OutputStream << "\"cat\":\"function\",";
    m_OutputStream << "\"dur\":" << (result.End - result.Start) << ',';
    m_OutputStream << "\"name\":\"" << name << "\",";
    m_OutputStream << "\"ph\":\"X\",";
    m_OutputStream << "\"pid\":0,";
    m_OutputStream << "\"tid\":" << result.ThreadID << ",";
    m_OutputStream << "\"ts\":" << result.Start;
    m_OutputStream << "}";

    m_OutputStream.flush();

    updateSummary(result);
  }

  void
  updateSummary(const ProfileResult & result)
  {
    m_SessionSummary[result.Name] += result.End - result.Start;
    m_TotalTime += result.End - result.Start;
  }

  void
  writeSummary()
  {
    if (m_SessionActive && !m_OutputStream.is_open())
    {
      m_OutputStream.open(m_CurrentSession->Filename + ".txt");
      std::multimap<long long, std::string, std::greater<long long>> sortedSummary = flipMap(m_SessionSummary);
      for (auto const & x : sortedSummary)
      {
        double pct = (double)x.first / (double)m_TotalTime * 100.0;
        m_OutputStream << x.second << " : " << x.first << " = " << pct << "(%)" << std::endl;
      }
      m_OutputStream.flush();
      m_OutputStream.close();
    }
  }

  void
  writeHeader()
  {
    m_OutputStream << "{\"otherData\": {},\"traceEvents\":[";
    m_OutputStream.flush();
  }

  void
  writeFooter()
  {
    m_OutputStream << "]}";
    m_OutputStream.flush();
  }

  InstrumentationSession *         m_CurrentSession;
  std::ofstream                    m_OutputStream;
  int                              m_ProfileCount;
  bool                             m_SessionActive;
  std::map<std::string, long long> m_SessionSummary;
  long long                        m_TotalTime;
};

class InstrumentationTimer
{
  ALIAS_TEMPLATE_FUNC(getTimePoint, std::chrono::time_point_cast<std::chrono::microseconds>)

public:
  InstrumentationTimer(const char * name)
    : m_Name(name)
    , m_Stopped(false)
  {
    m_StartTimepoint = std::chrono::high_resolution_clock::now();
  }

  ~InstrumentationTimer()
  {
    if (!m_Stopped)
      stop();
  }

  void
  stop()
  {
    auto endTimepoint = std::chrono::high_resolution_clock::now();

    long long start = getTimePoint(m_StartTimepoint).time_since_epoch().count();
    long long end = getTimePoint(endTimepoint).time_since_epoch().count();

    uint32_t threadID = std::hash<std::thread::id>{}(std::this_thread::get_id());
    Instrumentor::write({ m_Name, start, end, threadID });

    m_Stopped = true;
  }

private:
  const char *                                                m_Name;
  std::chrono::time_point<std::chrono::high_resolution_clock> m_StartTimepoint;
  bool                                                        m_Stopped;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ELASTIX_ENABLE_PROFILING
#  define PROFILE_BEGIN_SESSION(name, filename) Instrumentor::beginSession(name, filename)
#  define PROFILE_END_SESSION() Instrumentor::endSession()
#  define PROFILE_SCOPE(name) InstrumentationTimer CONCAT(timer, __LINE__)(name)
#  define PROFILE_FUNCTION() PROFILE_SCOPE(__PRETTY_FUNCTION__)
#else
#  define PROFILE_BEGIN_SESSION(name, filepath)
#  define PROFILE_END_SESSION()
#  define PROFILE_SCOPE(name)
#  define PROFILE_FUNCTION()
#endif