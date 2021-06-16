// Copyright 2016 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.!

#ifndef NORMALIZER_CASE_ENCODER_H_
#define NORMALIZER_CASE_ENCODER_H_

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <deque>

#include "common.h"
#include "third_party/absl/strings/string_view.h"


namespace sentencepiece {
namespace normalizer {

std::vector<std::pair<const char*, const char*>> search(const std::string& input);

constexpr char cUppercase    = 'U';
constexpr char cAllUppercase = 'A';
constexpr char cTitlecase    = 'T';
constexpr char cLowercase    = 'L';
constexpr char cPunctuation  = 'P';
constexpr char cSpace        = ' ';

class CaseEncoder {
protected:
  typedef std::function<std::pair<absl::string_view, int>(absl::string_view)> Normalizer;
  Normalizer normalizer_;

public:
  virtual ~CaseEncoder() {}
  
  virtual std::pair<absl::string_view, int> normalizePrefix(absl::string_view input) {
    return normalizer_(input);
  }

  virtual void setNormalizer(Normalizer normalizer) {
    normalizer_ = normalizer;
  }

  virtual void postProcess(std::string* normalized, std::vector<size_t>* norm_to_orig) {}

  static std::unique_ptr<CaseEncoder> Create(bool, bool, bool);
};

class UpperCaseEncoder : public CaseEncoder {
private:
  std::string buffer_;
  std::string signature_;
  
  int state_{0};
  size_t spans_{0};
  bool seenThreeSpans_{false};
  bool removeExtraWhiteSpace_{false};

public:
  UpperCaseEncoder(bool removeExtraWhiteSpace)
  : removeExtraWhiteSpace_(removeExtraWhiteSpace) {}

  std::pair<absl::string_view, int> normalizePrefix(absl::string_view input) {
    auto p = CaseEncoder::normalizePrefix(input);
    auto sp = p.first;
    int consumed = p.second;

    bool last = input.size() == (size_t)consumed;
    decltype(p) ret;

    auto null = [](int consumed) -> std::pair<absl::string_view, int> {
      return {{nullptr, 0}, consumed};
    };

    auto buffer = [this](absl::string_view sp) {
      buffer_.append(sp.data(), sp.size());
    };

    auto isUpper  = [=](absl::string_view sp) { return sp[0] == cUppercase;   };
    auto isPunct  = [=](absl::string_view sp) { return sp[0] == cPunctuation; };
    auto isSpace  = [=](absl::string_view sp) { return sp[0] == ' '; };

    if(state_ == 0)
      buffer_.clear();

    if(isUpper(sp)) {
      if(state_ == 0) {
        buffer(sp);
        
        buffer_[0] = cTitlecase;
        state_ = 1;
        ret = null(consumed);
        
        signature_.append("U");
        signature_.append(sp.size() - 1, 'u');

      } else if(state_ == 1 || state_ == 2) {
        if(state_ == 1)
          spans_++;
        
        sp.remove_prefix(1);
        buffer(sp);

        buffer_[0] = cUppercase;
        state_ = 2;
        ret = null(consumed);

        signature_.append(sp.size(), 'u');
      }  

      if(last)
        ret.first = absl::string_view(buffer_);

    } else {
      if(isPunct(sp)) {
        if(state_ == 1)
          spans_++;

        sp.remove_prefix(1);
        signature_.append(sp.size(), 'p');
      } else if(state_ == 2 && !isSpace(sp)) {
        spans_ = 0;
        buffer_ += cLowercase;
        signature_.append("L");
        signature_.append(sp.size(), 'l');
      } else if(isSpace(sp)) {
        if(state_ == 1)
          spans_++;
        if(!removeExtraWhiteSpace_ || signature_.empty() || signature_.back() != 's')
          signature_.append("sss");
      } else {
        spans_ = 0;
        signature_.append(sp.size(), 'l');
      }

      if(!buffer_.empty()) {
        buffer(sp);
        p.first = absl::string_view(buffer_);
      } else {
        p.first = sp;
      }
      
      state_ = 0;
      ret = p;
    }

    if(spans_ >= 3)
      seenThreeSpans_ = true;

    return ret;
  }

  virtual void postProcess(std::string* normalized, std::vector<size_t>* norm_to_orig) { 
    if(!seenThreeSpans_)
      return;

    std::string normalized_temp;
    normalized_temp.reserve(normalized->size());
    
    std::vector<size_t> norm_to_orig_temp;
    norm_to_orig_temp.reserve(norm_to_orig->size());

    const char* sig_it = signature_.data();

    auto nrm_it = normalized->cbegin();
    auto n2o_it = norm_to_orig->cbegin();

    for(const auto& span : search(signature_)) {
      size_t len = std::distance(sig_it, span.first);
      
      normalized_temp.insert(normalized_temp.end(), nrm_it, nrm_it + len);
      norm_to_orig_temp.insert(norm_to_orig_temp.end(), n2o_it, n2o_it + len);

      sig_it += len; 
      nrm_it += len;
      n2o_it += len;
      normalized_temp.push_back(cAllUppercase);
      norm_to_orig_temp.push_back(*n2o_it);
            
      while(sig_it != span.second) {
        if(*sig_it == cUppercase) {
          sig_it++; 
          nrm_it++;
          n2o_it++;
        }
        sig_it++;
        normalized_temp.push_back(*nrm_it++);
        norm_to_orig_temp.push_back(*n2o_it++);
      }
      if(sig_it != signature_.data() + signature_.length()) { 
        if(*sig_it != cUppercase) {
          normalized_temp.push_back(cLowercase);
          norm_to_orig_temp.push_back(*n2o_it);
        }
      }
    }

    if(nrm_it != normalized->cend())
      normalized_temp.insert(normalized_temp.end(), nrm_it, normalized->cend());
    if(n2o_it != norm_to_orig->cend())
      norm_to_orig_temp.insert(norm_to_orig_temp.end(), n2o_it, norm_to_orig->cend());

    normalized->swap(normalized_temp);
    norm_to_orig->swap(norm_to_orig_temp);
  }
};

class UpperCaseDecoder : public CaseEncoder {
private:
  std::unique_ptr<std::string> buffer_;
  absl::string_view input_;

  int state_ = 0;
  bool allUp_{false};

public:
  UpperCaseDecoder() {}

  std::pair<absl::string_view, int> normalizePrefix(absl::string_view input) {
    if(!buffer_) {
      buffer_.reset(new std::string(input.data(), input.size()));
      input_ = absl::string_view(*buffer_);
    }

    if(input_[0] == cAllUppercase) {
      const_cast<char&>(input_[0]) = cUppercase;
      allUp_ = true;
    } else if (input_[0] == cTitlecase) {
      allUp_ = false;
    } else if (input_[0] == cLowercase) {
      allUp_ = false;
    }

    auto p = CaseEncoder::normalizePrefix(input_);
    int consumed = p.second;

    if(input_[0] == cUppercase) {
      if(state_ == 0) { 
        input_.remove_prefix(consumed - 1);
        const_cast<char&>(input_[0]) = cUppercase;
        state_ = 1;
      } else if(state_ == 1) {
        if(consumed > 1) {
          input_.remove_prefix(consumed - 1);
          const_cast<char&>(input_[0]) = cUppercase;
          p.second = consumed - 1;
          state_ = 1;
        } else {
          input_.remove_prefix(consumed);
          p.first.remove_prefix(1);
          p.second = 0;
          state_ = 0;
        }
      }
    } else if(input_[0] == cLowercase) {
      input_.remove_prefix(consumed);
      p.first.remove_prefix(1);
      state_ = 0;
    } else {
      if(allUp_) {
        p.first = absl::string_view(input.data(), p.first.size());
        input_.remove_prefix(consumed - 1);
        const_cast<char&>(input_[0]) = cUppercase;
        state_ = 1;
      } else {
        input_.remove_prefix(consumed);
        state_ = 0;
      }
    }

    return p;
  }
};

}  // namespace normalizer
}  // namespace sentencepiece
#endif  // NORMALIZER_CASE_ENCODER_H_
