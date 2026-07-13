How to generate upb stub files:

1. Install protobuf compiler (protoc) and upb compiler (upbc).
   See: https://github.com/protocolbuffers/upb

2. Run the following commands to generate stub files from .proto files:

   # From this directory:
   protoc -I../ --upb_out=. ../sentencepiece.proto
   protoc -I../ --upb_out=. ../sentencepiece_model.proto

   # If you need upbdefs (not used by default, but kept for compatibility):
   # protoc -I../ --upbdefs_out=. ../sentencepiece.proto
   # protoc -I../ --upbdefs_out=. ../sentencepiece_model.proto

   (Note: depending on the upb version, you may need to specify the plugin path:
    --plugin=protoc-gen-upb=/path/to/upbc)

These generated files (*.upb.h, *.upb.c) are checked into VCS
to allow building without upb compiler dependency.


Differences from Standard Protobuf C++ API
===========================================

This UPB wrapper (`upb_wrapper.h`) emulates the standard Protobuf C++ API,
but there are several structural and behavioral differences due to the
underlying arena-based memory model of UPB:

1. Memory Lifecycle (Arena-based vs Object-based):
   - In standard Protobuf, each message object manages its own memory
     independently.
   - In this UPB wrapper, all sub-message elements share the memory pool
     allocated on the root message's `upb_Arena`.
   - Modifying a child wrapper object directly updates the raw memory on the
     arena. To ensure consistency, a callback mechanism (`on_change_`) is used
     to notify parent wrappers so they can rebuild or update their C++ wrapper
     caches.

2. Mutable Accessors (`mutable_xxx()`):
   - In standard Protobuf, `mutable_xxx()` always returns a pointer to the
     sub-message.
   - In this wrapper, calling `mutable_xxx()` for the first time dynamically
     allocates the sub-message structure on the arena and sets it onto the
     parent message, triggering cache synchronization.

3. Repeated Fields Operations:
   - Unlike standard Protobuf which exposes `RepeatedPtrField` or
     `RepeatedField` directly, UPB wrapper restricts repeated array operations
     to specific helper methods (`add_xxx()`, `set_xxx_at()`, `clear_xxx()`)
     due to UPB's C-API array limitations.

4. Thread Safety (Eager Cache Loading):
   - Standard Protobuf guarantees thread safety for concurrent read access
     (const getters).
   - To achieve the same thread safety in the UPB wrapper under concurrent
     inference, we completely eliminated Lazy-Initialization from all `const`
     getters (e.g. `pieces() const`).
   - Instead, caches are eagerly initialized in `OnArenaReset()` right after
     parsing or loading a model. Lazy-Init is only retained in non-const getters
     for manual model construction in single-threaded environments.

5. Out-of-Bounds (OOB) Protection:
   - Standard Protobuf typically crashes (assertion failure or SegFault) when
     accessing index out-of-bounds on repeated fields.
   - This wrapper provides safer OOB protection: accessing invalid indices
     (e.g. `-1` or `size`) via `pieces(i)` or `nbests(i)` returns a static
     default empty instance or fallback default value rather than crashing.

6. Reflection and Metadata:
   - Dynamic reflection APIs (`GetDescriptor()`, `GetReflection()`) are not
     supported. Only static accessors required by SentencePiece are
     implemented.


Anti-Patterns (NG Examples)
============================

Avoid the following patterns to prevent crashes, memory corruption, or race
conditions:

1. Dangling Sub-message Reference (Dangling Pointer):
   Never keep or use references to child wrappers (like `NormalizerSpec` or
   `TrainerSpec`) after the parent `ModelProto` is destroyed. Sub-messages
   share the parent's `upb_Arena` lifetime.

   // --- NG EXAMPLE ---
   const NormalizerSpec* spec = nullptr;
   {
     ModelProto model;
     model.ParseFromString(data);
     spec = &model.normalizer_spec(); // Holds reference to the arena memory
   } // 'model' is destroyed here, freeing the arena.
   
   std::cout << spec->name(); // CRASH! Use-after-free on deleted arena.


2. Concurrent Access to Non-const or Mutable Getters:
   While `const` getters (e.g. `pieces(i) const`) are thread-safe for
   concurrent read access during inference, non-const getters (e.g. `pieces(i)`)
   and mutable accessors (e.g. `mutable_pieces(idx)` or
   `mutable_normalizer_spec()`) are not thread-safe and will cause Data Races or
   memory corruption.

   // --- NG EXAMPLE (Multi-threaded context) ---
   void ThreadWorker(ModelProto* shared_model) {
     // NG: Using mutable or non-const accessors concurrently triggers internal
     // memory allocation on the arena and cache synchronization callbacks.
     auto* spec = shared_model->mutable_normalizer_spec();
     spec->set_add_dummy_prefix(false);
   }
   
   // --- CORRECT WAY ---
   void ThreadWorker(const ModelProto* shared_model) {
     // OK: Const accessor is read-only and completely thread-safe.
     const auto& spec = shared_model->normalizer_spec();
     std::cout << spec.add_dummy_prefix();
   }


3. Out-of-bounds Array Mutation:
   Do not modify repeated fields using out-of-bounds indices via raw setters
   like `set_piece_at`. You must first allocate the element using `add_xxx()`.

   // --- NG EXAMPLE ---
   ModelProto model; // 'pieces' size is 0
   model.set_piece_at(5, "token"); // CRASH! Out-of-bounds array write.

   // --- CORRECT WAY ---
   auto* piece = model.add_pieces(); // Allocates on the arena and increments size.
   piece->set_piece("token");


4. Excessive Copy Assignment in Loops (Arena Bloat):
   Assigning wrappers using `operator=` triggers serialization and re-allocation
   on the parent's arena. Avoid repeated copy assignments inside loops.

   // --- NG EXAMPLE ---
   ModelProto model;
   NormalizerSpec local_spec;
   local_spec.set_name("nfkc");
   
   for (int i = 0; i < 1000; ++i) {
     // NG: Allocates new memory on the parent arena during each assignment, 
     // leading to memory bloat because arena memory is only freed as a whole.
     *model.mutable_normalizer_spec() = local_spec; 
   }

   // --- CORRECT WAY ---
   // Modify directly via mutable accessor to avoid temporary allocations.
   model.mutable_normalizer_spec()->set_name("nfkc");
