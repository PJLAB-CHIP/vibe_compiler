//===- SimpleRoute.h - Lazy simple physical routes -----------*- C++ -*-===//

#pragma once

#include "Wafer/IR/Target/TargetTopology.h"

#include <optional>

namespace wafer::compiler::detail::simple_route {

struct RouteSuccessor {
  bool currentFound = false;
  std::optional<llvm::SmallVector<TileLink, 8>> next;
};

bool isValidSimpleRoute(const TargetTopology &topology, CardId cardId,
                        TileId source, TileId destination,
                        llvm::ArrayRef<TileLink> route);

std::optional<llvm::SmallVector<TileLink, 8>>
getFirstSimpleRoute(const TargetTopology &topology, CardId cardId,
                    TileId source, TileId destination);

RouteSuccessor getNextSimpleRoute(const TargetTopology &topology, CardId cardId,
                                  TileId source, TileId destination,
                                  llvm::ArrayRef<TileLink> currentRoute);

} // namespace wafer::compiler::detail::simple_route
