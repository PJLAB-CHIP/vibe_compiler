// RUN: wafer-opt --wafer-convert-stablehlo-attention-to-linalg --split-input-file --verify-diagnostics %s -o /dev/null
// REQUIRES: stablehlo

// Each verifier-negative isolates one malformed semantic field. Real-scale
// source-to-structured positive coverage is in wafer-frontend-product.test.
module {
  func.func private @body(%q: tensor<1x4x1025x64xf16>, %k: tensor<1x2x1025x64xf16>, %v: tensor<1x2x1025x64xf16>) -> tensor<1x4x1025x64xf16> { return %q : tensor<1x4x1025x64xf16> }
  func.func @missing_position(%q: tensor<1x4x1025x64xf16>, %k: tensor<1x2x1025x64xf16>, %v: tensor<1x2x1025x64xf16>) -> tensor<1x4x1025x64xf16> {
    // expected-error@+1 {{unsupported or incomplete scaled dot-product attention composite contract}}
    %o = stablehlo.composite "wafer.scaled_dot_product_attention" %q, %k, %v {composite_attributes = {is_causal = true, has_mask = false, enable_gqa = true, scale = 0.125 : f64, query_start = 0 : i64, key_start = 0 : i64}, decomposition = @body} : (tensor<1x4x1025x64xf16>, tensor<1x2x1025x64xf16>, tensor<1x2x1025x64xf16>) -> tensor<1x4x1025x64xf16>
    return %o : tensor<1x4x1025x64xf16>
  }
}

// -----

module {
  func.func private @body(%q: tensor<1x4x1025x64xf16>, %k: tensor<1x2x1025x64xf16>, %v: tensor<1x2x1025x64xf16>) -> tensor<1x4x1025x64xf16> { return %q : tensor<1x4x1025x64xf16> }
  func.func @gqa_disabled(%q: tensor<1x4x1025x64xf16>, %k: tensor<1x2x1025x64xf16>, %v: tensor<1x2x1025x64xf16>) -> tensor<1x4x1025x64xf16> {
    // expected-error@+1 {{unsupported or incomplete scaled dot-product attention composite contract}}
    %o = stablehlo.composite "wafer.scaled_dot_product_attention" %q, %k, %v {composite_attributes = {is_causal = true, has_mask = false, enable_gqa = false, scale = 0.125 : f64, query_start = 0 : i64, key_start = 0 : i64, key_valid_end = 1025 : i64}, decomposition = @body} : (tensor<1x4x1025x64xf16>, tensor<1x2x1025x64xf16>, tensor<1x2x1025x64xf16>) -> tensor<1x4x1025x64xf16>
    return %o : tensor<1x4x1025x64xf16>
  }
}

// -----

module {
  func.func private @body(%q: tensor<1x4x1025x64xf16>, %k: tensor<1x2x1025x64xf16>, %v: tensor<1x2x1025x64xf16>) -> tensor<1x4x1025x64xf16> { return %q : tensor<1x4x1025x64xf16> }
  func.func @negative_query(%q: tensor<1x4x1025x64xf16>, %k: tensor<1x2x1025x64xf16>, %v: tensor<1x2x1025x64xf16>) -> tensor<1x4x1025x64xf16> {
    // expected-error@+1 {{unsupported or incomplete scaled dot-product attention composite contract}}
    %o = stablehlo.composite "wafer.scaled_dot_product_attention" %q, %k, %v {composite_attributes = {is_causal = true, has_mask = false, enable_gqa = true, scale = 0.125 : f64, query_start = -1 : i64, key_start = 0 : i64, key_valid_end = 1025 : i64}, decomposition = @body} : (tensor<1x4x1025x64xf16>, tensor<1x2x1025x64xf16>, tensor<1x2x1025x64xf16>) -> tensor<1x4x1025x64xf16>
    return %o : tensor<1x4x1025x64xf16>
  }
}

// -----

module {
  func.func private @body(%q: tensor<1x4x1025x64xf16>, %k: tensor<1x2x1025x64xf16>, %v: tensor<1x2x1025x64xf16>) -> tensor<1x4x1025x64xf16> { return %q : tensor<1x4x1025x64xf16> }
  func.func @invalid_end(%q: tensor<1x4x1025x64xf16>, %k: tensor<1x2x1025x64xf16>, %v: tensor<1x2x1025x64xf16>) -> tensor<1x4x1025x64xf16> {
    // expected-error@+1 {{unsupported or incomplete scaled dot-product attention composite contract}}
    %o = stablehlo.composite "wafer.scaled_dot_product_attention" %q, %k, %v {composite_attributes = {is_causal = true, has_mask = false, enable_gqa = true, scale = 0.125 : f64, query_start = 0 : i64, key_start = 0 : i64, key_valid_end = 1026 : i64}, decomposition = @body} : (tensor<1x4x1025x64xf16>, tensor<1x2x1025x64xf16>, tensor<1x2x1025x64xf16>) -> tensor<1x4x1025x64xf16>
    return %o : tensor<1x4x1025x64xf16>
  }
}

// -----

module {
  func.func private @body(%q: tensor<1x4x1025x64xf16>, %k: tensor<1x2x1025x64xf16>, %v: tensor<1x2x1025x64xf16>) -> tensor<1x4x1025x64xf16> { return %q : tensor<1x4x1025x64xf16> }
  func.func @wrong_position_type(%q: tensor<1x4x1025x64xf16>, %k: tensor<1x2x1025x64xf16>, %v: tensor<1x2x1025x64xf16>) -> tensor<1x4x1025x64xf16> {
    // expected-error@+1 {{unsupported or incomplete scaled dot-product attention composite contract}}
    %o = stablehlo.composite "wafer.scaled_dot_product_attention" %q, %k, %v {composite_attributes = {is_causal = true, has_mask = false, enable_gqa = true, scale = 0.125 : f64, query_start = 0.0 : f64, key_start = 0 : i64, key_valid_end = 1025 : i64}, decomposition = @body} : (tensor<1x4x1025x64xf16>, tensor<1x2x1025x64xf16>, tensor<1x2x1025x64xf16>) -> tensor<1x4x1025x64xf16>
    return %o : tensor<1x4x1025x64xf16>
  }
}

// -----

module {
  func.func private @body(%q: tensor<1x4x1025x64xf16>, %k: tensor<1x2x1025x64xf16>, %v: tensor<1x2x1025x64xf16>) -> tensor<1x4x1025x64xf16> { return %q : tensor<1x4x1025x64xf16> }
  func.func @unknown_field(%q: tensor<1x4x1025x64xf16>, %k: tensor<1x2x1025x64xf16>, %v: tensor<1x2x1025x64xf16>) -> tensor<1x4x1025x64xf16> {
    // expected-error@+1 {{unsupported or incomplete scaled dot-product attention composite contract}}
    %o = stablehlo.composite "wafer.scaled_dot_product_attention" %q, %k, %v {composite_attributes = {is_causal = true, has_mask = false, enable_gqa = true, scale = 0.125 : f64, future_policy = true, query_start = 0 : i64, key_start = 0 : i64, key_valid_end = 1025 : i64}, decomposition = @body} : (tensor<1x4x1025x64xf16>, tensor<1x2x1025x64xf16>, tensor<1x2x1025x64xf16>) -> tensor<1x4x1025x64xf16>
    return %o : tensor<1x4x1025x64xf16>
  }
}

// -----

module {
  func.func private @body(%q: tensor<1x4x1025x64xf16>, %k: tensor<1x2x1025x64xf16>, %v: tensor<1x2x1025x64xf16>) -> tensor<1x4x1025x64xf16> { return %q : tensor<1x4x1025x64xf16> }
  func.func @missing_mask(%q: tensor<1x4x1025x64xf16>, %k: tensor<1x2x1025x64xf16>, %v: tensor<1x2x1025x64xf16>) -> tensor<1x4x1025x64xf16> {
    // expected-error@+1 {{unsupported or incomplete scaled dot-product attention composite contract}}
    %o = stablehlo.composite "wafer.scaled_dot_product_attention" %q, %k, %v {composite_attributes = {is_causal = true, has_mask = true, enable_gqa = true, scale = 0.125 : f64, query_start = 0 : i64, key_start = 0 : i64, key_valid_end = 1025 : i64}, decomposition = @body} : (tensor<1x4x1025x64xf16>, tensor<1x2x1025x64xf16>, tensor<1x2x1025x64xf16>) -> tensor<1x4x1025x64xf16>
    return %o : tensor<1x4x1025x64xf16>
  }
}

// -----

module {
  func.func private @body(%q: tensor<1x4x1025x64xf16>, %k: tensor<1x2x1025x64xf16>, %v: tensor<1x2x1025x64xf16>) -> tensor<1x4x1025x64xf16> { return %q : tensor<1x4x1025x64xf16> }
  func.func @position_integer_overflow(%q: tensor<1x4x1025x64xf16>, %k: tensor<1x2x1025x64xf16>, %v: tensor<1x2x1025x64xf16>) -> tensor<1x4x1025x64xf16> {
    // expected-error@+1 {{unsupported or incomplete scaled dot-product attention composite contract}}
    %o = stablehlo.composite "wafer.scaled_dot_product_attention" %q, %k, %v {composite_attributes = {is_causal = true, has_mask = false, enable_gqa = true, scale = 0.125 : f64, query_start = 18446744073709551616 : i128, key_start = 0 : i64, key_valid_end = 1025 : i64}, decomposition = @body} : (tensor<1x4x1025x64xf16>, tensor<1x2x1025x64xf16>, tensor<1x2x1025x64xf16>) -> tensor<1x4x1025x64xf16>
    return %o : tensor<1x4x1025x64xf16>
  }
}
