set(WAFER_COMPILER_TRANSPORT_SOURCES
  Transport/DirectDTETransport.cpp
  PARENT_SCOPE)

set(WAFER_COMPILER_TRANSPORT_OPTIONAL_SOURCES
  Transport/CoordinatedCommunicationAction.cpp
  Transport/NoCCommunicationAction.cpp
  Transport/NoCIntermediateRouting.cpp
  Transport/NoCPartialDataflow.cpp
  PARENT_SCOPE)
