// ...

void Normalizer::Init() {
  // ...

  if (!index.empty()) {
    // ...

    trie_ = std::make_unique<Darts::DoubleArray>();

    // ...

    if (!trie_->validate()) {
      status_ = util::InternalError(
          "Trie data contains out-of-bounds node references.");
      return;
    }

    // Check the root node's offset
    if (trie_->array_[0].offset() >= trie_->size_) {
      status_ = util::InternalError(
          "Trie data contains out-of-bounds node references.");
      return;
    }
  }
}

// ...