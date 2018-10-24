#include <limits>
#include <sstream>

#include "utils/inc/asserter.hpp"
#include "utils/inc/debug.hpp"

#include "arch/inc/arch.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/waveop.hpp"
#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/sbatomloadwaveop.hpp"
//#include "wave/inc/sbatomsavewaveop.hpp"
//#include "wave/inc/barrierwaveop.hpp"
#include "wave/inc/nopwaveop.hpp"

#include "nets/inc/network.hpp"

#include "events/inc/events.hpp"
#include "events/inc/eventmgr.hpp"


namespace kcc {
namespace events {



/***************************************************************
***************************************************************/
void
EventMgr::EventState::mvEventFromSetToSet(EventId evtId, EventSet& fromSet, EventSet& toSet,
        const char* fromStr, const char* toStr)
{
    Assert(qEventRegular(evtId), "Cannot move non-regular event id from ", fromStr, " to ", toStr);
    Assert(fromSet.find(evtId) != fromSet.end(), "Event from prev edge not in ", fromStr);
    Assert(toSet.find(evtId) == toSet.end(), "Event from prev edge already in the ", toStr, " set");
    fromSet.erase(evtId);
    toSet.insert(evtId);
}

/***********************************************************************
***********************************************************************/
void
EventMgr::EventState::mvFromInFlightToCompleted(EventId evtId)
{
    mvEventFromSetToSet(evtId, m_InFlight, m_Completed, "InFlight", "Completed");
}


/***********************************************************************
***********************************************************************/
void
EventMgr::EventState::mvFromAvailableToInFlight(EventId evtId)
{
    mvEventFromSetToSet(evtId, m_Available, m_InFlight, "Available", "InFlight");
}

void
EventMgr::EventState::mvFromCompletedToAvailable(EventId evtId)
{
    mvEventFromSetToSet(evtId, m_Completed, m_Available, "Completed", "Available");
}




/***********************************************************************
***********************************************************************/
void
EventMgr::EventState::clearAll()
{
    m_Available.clear();
    m_InFlight.clear();
    m_Completed.clear();
}

/***********************************************************************
***********************************************************************/
void
EventMgr::EventState::clearCompleted()
{
    m_Completed.clear();
}


/***********************************************************************
***********************************************************************/
void
EventMgr::EventState::moveCompletedEventsToAvailable()
{
    // Avaliable += Completed;
    for (auto evtId : m_Completed) {
        const auto ret = addAvailable(evtId); // ret.second is false if element already exists
        Assert(ret.second, "Event id ", evtId, " already in completed and available event sets");
    }
    clearCompleted();
}


/***********************************************************************
***********************************************************************/
void
EventMgr::EventState::init()
{
    clearAll();

    for (EventId eventId = ReservedEvent_FirstNonReserved;
         eventId <= EventId_LastNonReserved(); ++eventId)
    {
        addAvailable(eventId);
    }
}

}}

