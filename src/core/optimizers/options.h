#pragma once
// ─────────────────────────────────────────────────────────────────────────────
//  options.h — NeuralStudio compatibility shim.
//
//  In the original optimsolution framework this header defines structures for
//  local-search (L-BFGS etc.) options.  NeuralStudio stubs all local-search
//  hooks to no-ops, so no types from this header are actually used.
//  The empty include satisfies the #include directive in cmaes.cpp / lmcmaes.cpp.
// ─────────────────────────────────────────────────────────────────────────────
