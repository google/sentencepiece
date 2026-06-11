// ...

bool DoubleArrayImpl::validate() const {
  for (int i = 0; i < size_; ++i) {
    if (array_[i].label() > 0xFF) {
      // Check the offset even if the label has bit 31 set
      if (array_[i].offset() >= size_) {
        return false;
      }
    }
  }
  return true;
}

// ...