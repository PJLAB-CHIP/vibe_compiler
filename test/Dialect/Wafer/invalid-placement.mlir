// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module attributes {
  wafer.placement = #wafer.placement<tile>
} {}

// CHECK: expected ::wafer::Placement to be one of: single_tile, tile_mesh, card_mesh
