// RUN: wafer-opt -split-input-file -verify-diagnostics %s

module {
  func.func @accepts_external_instrumentation() {
    "wafer.tile.region"() ({
      "wafer.tile.yield"() : () -> ()
    }) {test.instrumentation = "enabled"} : () -> ()
    return
  }
}

// -----

module {
  func.func @rejects_schema_free_wafer_semantics() {
    // expected-error @+1 {{does not accept schema-free semantic attribute 'wafer.hidden_semantics'}}
    "wafer.tile.region"() ({
      "wafer.tile.yield"() : () -> ()
    }) {wafer.hidden_semantics = "forbidden"} : () -> ()
    return
  }
}
