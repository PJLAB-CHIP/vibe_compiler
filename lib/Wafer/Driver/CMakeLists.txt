add_subdirectory(StandaloneTileModules)

set(WAFER_DRIVER_SOURCES
  Driver/Compilation.cpp
  Driver/CompilationOrchestration.cpp
  Driver/CompilationStages.cpp
  Driver/ExecutableCompilation.cpp
  Driver/ExecutableLowering.cpp
  Driver/PhysicalDataflow/ActualResultController.cpp
  Driver/PhysicalDataflow/StructuredProgramAnalysis.cpp
  Driver/PhysicalDataflow/UnifiedSearch.cpp
  Driver/ProgramDirectoryTransaction.cpp
  Driver/ProgramData.cpp
  Driver/ProgramResourceVerification.cpp
  Driver/SpmdCompilationBridge.cpp
  Driver/TensorProgramCompilation.cpp
  PARENT_SCOPE)
