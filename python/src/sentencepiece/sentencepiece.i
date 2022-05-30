%module sentencepiece
%include exception.i

%{
#include <algorithm>
#include <limits>
#include <cmath>
#include <sentencepiece_processor.h>
#include <sentencepiece_trainer.h>

namespace {
PyObject* kUnicodeInput = reinterpret_cast<PyObject* >(0x1);
PyObject* kByteInput = reinterpret_cast<PyObject* >(0x2);

inline void ReleaseResultObject(PyObject *obj) {
  if (obj != nullptr && obj != kUnicodeInput && obj != kByteInput) {
    Py_XDECREF(obj);
  }
}

class PyInputString {
 public:
  explicit PyInputString(PyObject* obj) {
    if (PyUnicode_Check(obj)) {
      str_ = const_cast<char *>(PyUnicode_AsUTF8AndSize(obj, &size_));
      input_type_ = kUnicodeInput;
    } else if (PyBytes_Check(obj)) {
      PyBytes_AsStringAndSize(obj, &str_, &size_);
      input_type_ = kByteInput;
    } else {
      str_ = nullptr;
    }
  }
  const char* data() const { return str_; }
  Py_ssize_t size() const { return size_; }
  bool IsAvalable() const { return str_ != nullptr; }
  PyObject *input_type() const { return input_type_; }

  static bool IsUnicode(PyObject *resultobj) {
    return (resultobj == nullptr || resultobj == kUnicodeInput);
  }

 private:
  PyObject* input_type_ = nullptr;
  char* str_ = nullptr;
  Py_ssize_t size_ = 0;
};

PyObject* MakePyOutputString(const std::string& output,
                             PyObject *resultobj) {
  if (PyInputString::IsUnicode(resultobj)) {
    return PyUnicode_FromStringAndSize(output.data(), output.size());
  }
  return PyBytes_FromStringAndSize(output.data(), output.size());
}

PyObject* MakePyOutputBytes(const std::string& output) {
  return PyBytes_FromStringAndSize(output.data(), output.size());
}

int ToSwigError(sentencepiece::util::StatusCode code) {
  switch (code) {
    case sentencepiece::util::StatusCode::kNotFound:
      return SWIG_IOError;
    case sentencepiece::util::StatusCode::kOutOfRange:
      return SWIG_IndexError;
    case sentencepiece::util::StatusCode::kInvalidArgument:
      return SWIG_SyntaxError;
    default:
      return SWIG_RuntimeError;
  }
  return SWIG_RuntimeError;
}

class PySentenceIterator : public sentencepiece::SentenceIterator {
  public:
  PySentenceIterator(PyObject *iter) : iter_(iter) {
    item_ = PyIter_Next(iter_);
    CopyValue();
  }

  ~PySentenceIterator() {
   // Py_XDECREF(iter_);
  }

  bool done() const override {
    return item_ == nullptr;
  }

  void Next() override {
    item_ = PyIter_Next(iter_);
    CopyValue();
  }

  const std::string &value() const override {
    return value_;
  }

  sentencepiece::util::Status status() const override {
    return status_;
  }

  private:
   void CopyValue() {
     if (item_ == nullptr) return;
     const PyInputString ustring(item_);
     if (ustring.IsAvalable()) {
       const char *data = ustring.data();
       size_t size = ustring.size();
       while (size > 0) {
         if (data[size - 1] == '\r' || data[size - 1] == '\n')
           --size;
         else
           break;
       }
       value_.assign(data, size);
     } else {
       status_ = sentencepiece::util::Status(sentencepiece::util::StatusCode::kInternal,
                                             "Not a string.");
     }
     Py_XDECREF(item_);
   }
   PyObject *iter_ = nullptr;
   PyObject *item_ = nullptr;
   std::string value_;
   sentencepiece::util::Status status_;
};

void RewriteIds(const sentencepiece::SentencePieceProcessor &sp,
                std::vector<int> *ids,
                bool add_bos, bool add_eos, bool reverse) {
  if (!add_bos && !add_eos && !reverse) return;
  if (reverse) std::reverse(ids->begin(), ids->end());
  if (add_bos) ids->insert(ids->begin(), sp.bos_id());
  if (add_eos) ids->push_back(sp.eos_id());
}

void RewritePieces(const sentencepiece::SentencePieceProcessor &sp,
                   std::vector<std::string> *pieces,
                   bool add_bos, bool add_eos, bool reverse, bool emit_unk_piece) {
  if (!add_bos && !add_eos && !reverse && !emit_unk_piece) return;
  if (reverse) std::reverse(pieces->begin(), pieces->end());
  if (add_bos) pieces->insert(pieces->begin(), sp.IdToPiece(sp.bos_id()));
  if (add_eos) pieces->push_back(sp.IdToPiece(sp.eos_id()));
  if (emit_unk_piece) {
    const auto &unk = sp.IdToPiece(sp.unk_id());
    for (auto &piece : *pieces) {
      const int id = sp.PieceToId(piece);
      if (id == sp.unk_id()) {
        piece = unk;
      }
    }
  }
}
}  // namespace
%}

%exception {
  try {
    $action
    ReleaseResultObject(resultobj);
  }
  catch (const sentencepiece::util::Status &status) {
    SWIG_exception(ToSwigError(status.code()), status.ToString().c_str());
  }
}

%ignore sentencepiece::util::Status;
%ignore sentencepiece::util::StatusCode;
%ignore absl::string_view;
%ignore sentencepiece::SentencePieceText;
%ignore sentencepiece::NormalizerSpec;
%ignore sentencepiece::TrainerSpec;

%ignore sentencepiece::SentencePieceProcessor::status;
%ignore sentencepiece::SentencePieceProcessor::Encode;
%ignore sentencepiece::SentencePieceProcessor::SampleEncode;
%ignore sentencepiece::SentencePieceProcessor::NBestEncode;
%ignore sentencepiece::SentencePieceProcessor::SampleEncodeAndScore;
%ignore sentencepiece::SentencePieceProcessor::Decode;
%ignore sentencepiece::SentencePieceProcessor::DecodeIds;
%ignore sentencepiece::SentencePieceProcessor::DecodeIdsAsSerializedProto;
%ignore sentencepiece::SentencePieceProcessor::model_proto;
%ignore sentencepiece::SentencePieceProcessor::Load;
%ignore sentencepiece::SentencePieceProcessor::LoadOrDie;
%ignore sentencepiece::pretokenizer::PretokenizerForTrainingInterface;
%ignore sentencepiece::SentenceIterator;
%ignore sentencepiece::SentencePieceTrainer::Train;
%ignore sentencepiece::SentencePieceTrainer::GetNormalizerSpec;
%ignore sentencepiece::SentencePieceTrainer::PopulateNormalizerSpec;
%ignore sentencepiece::SentencePieceTrainer::MergeSpecsFromArgs;
%ignore sentencepiece::SentencePieceTrainer::SetProtoField;
%ignore sentencepiece::SentencePieceTrainer::PopulateModelTypeFromString;
%ignore sentencepiece::SentencePieceTrainer::PieceProcecssor;
%ignore sentencepiece::SentencePieceTrainer::SetPretokenizerForTraining;
%ignore sentencepiece::SentencePieceTrainer::GetPretokenizerForTraining;

%extend sentencepiece::SentencePieceProcessor {
  sentencepiece::util::Status LoadFromFile(absl::string_view arg) {
    return $self->Load(arg);
  }

  std::string DecodeIdsWithCheck(
      const std::vector<int> &ids) const {
    const int num_pieces = $self->GetPieceSize();
    for (int id : ids) {
      if (id < 0 || id >= num_pieces) {
        throw sentencepiece::util::Status(
            sentencepiece::util::StatusCode::kOutOfRange,
            "piece id is out of range.");
      }
    }
    return $self->DecodeIds(ids);
  }

  util::bytes DecodeIdsAsSerializedProtoWithCheck(
      const std::vector<int> &ids) const {
    const int num_pieces = $self->GetPieceSize();
    for (int id : ids) {
      if (id < 0 || id >= num_pieces) {
        throw sentencepiece::util::Status(
            sentencepiece::util::StatusCode::kOutOfRange,
            "piece id is out of range.");
      }
    }
    return $self->DecodeIdsAsSerializedProto(ids);
  }

  std::vector<int> _EncodeAsIds(absl::string_view text,
                                bool enabele_sampling,
                                int nbest_size, float alpha,
                                bool add_bos, bool add_eos, bool reverse) {
    auto ids = enabele_sampling ?
               $self->SampleEncodeAsIds(text, nbest_size, alpha) :
               $self->EncodeAsIds(text);
    RewriteIds(*$self, &ids, add_bos, add_eos, reverse);
    return ids;
  }

  std::vector<std::string> _EncodeAsPieces(absl::string_view text,
                                           bool enabele_sampling,
                                           int nbest_size, float alpha,
                                           bool add_bos, bool add_eos, bool reverse,
                                           bool emit_unk_piece) {
    auto pieces = enabele_sampling ?
                  $self->SampleEncodeAsPieces(text, nbest_size, alpha) :
                  $self->EncodeAsPieces(text);
    RewritePieces(*$self, &pieces, add_bos, add_eos, reverse, emit_unk_piece);
    return pieces;
  }

  std::vector<std::vector<int>>
      _NBestEncodeAsIds(absl::string_view text,
                        int nbest_size,
                        bool add_bos, bool add_eos, bool reverse) {
    auto idss = $self->NBestEncodeAsIds(text, nbest_size);
    for (auto &ids : idss) {
      RewriteIds(*$self, &ids, add_bos, add_eos, reverse);
    }
    return idss;
  }

  std::vector<std::vector<std::string>>
      _NBestEncodeAsPieces(absl::string_view text,
                           int nbest_size,
                           bool add_bos, bool add_eos, bool reverse,
                           bool emit_unk_piece) {
    auto piecess = $self->NBestEncodeAsPieces(text, nbest_size);
    for (auto &pieces : piecess) {
      RewritePieces(*$self, &pieces, add_bos, add_eos, reverse, emit_unk_piece);
    }
    return piecess;
  }

  std::vector<std::pair<std::vector<int>, float>>
      _SampleEncodeAndScoreAsIds(absl::string_view text,
                                 int num_samples, float theta, bool wor,
                                 bool include_best,
                                 bool add_bos, bool add_eos, bool reverse) {
    auto idss = $self->SampleEncodeAndScoreAsIds(text, num_samples,
                                                 theta, wor, include_best);
    for (auto &ids : idss) {
      RewriteIds(*$self, &ids.first, add_bos, add_eos, reverse);
    }
    return idss;
  }

  std::vector<std::pair<std::vector<std::string>, float>>  
      _SampleEncodeAndScoreAsPieces(absl::string_view text,
                                    int num_samples, float theta, bool wor,
                                    bool include_best,
                                    bool add_bos, bool add_eos, bool reverse,
                                    bool emit_unk_piece) {
    auto piecess = $self->SampleEncodeAndScoreAsPieces(text, num_samples,
                                                       theta, wor, include_best);
    for (auto &pieces : piecess) {
      RewritePieces(*$self, &pieces.first, add_bos, add_eos, reverse, emit_unk_piece);
    }
    return piecess;
  }      

%pythoncode {
  def Init(self,
           model_file=None,
           model_proto=None,
           out_type=int,
           add_bos=False,
           add_eos=False,
           reverse=False,
           emit_unk_piece=False,
           enable_sampling=False,
           nbest_size=-1,
           alpha=0.1):
    """Initialzie sentencepieceProcessor.

    Args:
      model_file: The sentencepiece model file path.
      model_proto: The sentencepiece model serialized proto.
      out_type: output type. int or str.
      add_bos: Add <s> to the result (Default = false)
      add_eos: Add </s> to the result (Default = false) <s>/</s> is added after
        reversing (if enabled).
      reverse: Reverses the tokenized sequence (Default = false)
      emit_unk_piece: Emits the unk literal string (Default = false)
      nbest_size: sampling parameters for unigram. Invalid for BPE-Dropout.
                  nbest_size = {0,1}: No sampling is performed.
                  nbest_size > 1: samples from the nbest_size results.
                  nbest_size < 0: assuming that nbest_size is infinite and samples
                    from the all hypothesis (lattice) using
                    forward-filtering-and-backward-sampling algorithm.
      alpha: Soothing parameter for unigram sampling, and dropout probability of
        merge operations for BPE-dropout.
    """

    _sentencepiece_processor_init_native(self)
    self._out_type = out_type
    self._add_bos = add_bos
    self._add_eos = add_eos
    self._reverse = reverse
    self._emit_unk_piece = emit_unk_piece
    self._enable_sampling = enable_sampling
    self._nbest_size = nbest_size
    self._alpha = alpha
    if model_file or model_proto:
      self.Load(model_file=model_file, model_proto=model_proto)


  def Encode(self,
             input,
             out_type=None,
             add_bos=None,
             add_eos=None,
             reverse=None,
             emit_unk_piece=None,
             enable_sampling=None,
             nbest_size=None,
             alpha=None):
    """Encode text input to segmented ids or tokens.

      Args:
      input: input string. accepsts list of string.
      out_type: output type. int or str.
      add_bos: Add <s> to the result (Default = false)
      add_eos: Add </s> to the result (Default = false) <s>/</s> is added after
        reversing (if enabled).
      reverse: Reverses the tokenized sequence (Default = false)
      emit_unk_piece: Emits the unk literal string (Default = false)
      nbest_size: sampling parameters for unigram. Invalid for BPE-Dropout.
                  nbest_size = {0,1}: No sampling is performed.
                  nbest_size > 1: samples from the nbest_size results.
                  nbest_size < 0: assuming that nbest_size is infinite and samples
                    from the all hypothesis (lattice) using
                    forward-filtering-and-backward-sampling algorithm.
      alpha: Soothing parameter for unigram sampling, and merge probability for
             BPE-dropout (probablity 'p' in BPE-dropout paper).
    """

    if out_type is None:
      out_type = self._out_type
    if add_bos is None:
      add_bos = self._add_bos
    if add_eos is None:
      add_eos = self._add_eos
    if reverse is None:
      reverse = self._reverse
    if emit_unk_piece is None:
      emit_unk_piece = self._emit_unk_piece
    if enable_sampling is None:
      enable_sampling = self._enable_sampling
    if nbest_size is None:
      nbest_size = self._nbest_size
    if alpha is None:
      alpha = self._alpha

    if enable_sampling == True and (nbest_size is None or nbest_size == 0 or
                                    nbest_size == 1 or alpha is None):
      raise RuntimeError(
          'When enable_sampling is True, We must specify "nbest_size > 1" or "nbest_size = -1", '
          'and "alpha". "nbest_size" is enabled only on unigram mode ignored in BPE-dropout. '
          'when "nbest_size = -1" , this method samples from all candidates on the lattice '
          'instead of nbest segmentations.'
      )

    def _encode(text):
      if out_type is int:
        return self._EncodeAsIds(text, enable_sampling, nbest_size,
                                 alpha, add_bos, add_eos, reverse)
      else:
        return self._EncodeAsPieces(text, enable_sampling, nbest_size,
                                    alpha, add_bos, add_eos, reverse, emit_unk_piece)

    if type(input) is list:
      return [_encode(n) for n in input]

    return _encode(input)


  def NBestEncode(self,
                  input,
                  out_type=None,
                  add_bos=None,
                  add_eos=None,
                  reverse=None,
                  emit_unk_piece=None,
                  nbest_size=None):
    """NBestEncode text input to segmented ids or tokens.

      Args:
      input: input string. accepsts list of string.
      out_type: output type. int or str.
      add_bos: Add <s> to the result (Default = false)
      add_eos: Add </s> to the result (Default = false) <s>/</s> is added after reversing (if enabled).
      reverse: Reverses the tokenized sequence (Default = false)
      emit_unk_piece: Emits the unk literal string (Default = false)
      nbest_size: nbest size
    """

    if out_type is None:
      out_type = self._out_type
    if add_bos is None:
      add_bos = self._add_bos
    if add_eos is None:
      add_eos = self._add_eos
    if reverse is None:
      reverse = self._reverse
    if emit_unk_piece is None:
      emit_unk_piece = self._emit_unk_piece
    if nbest_size is None:
      nbest_size = self._nbest_size

    if nbest_size <= 0:
      nbest_size=1

    def _encode(text):
      if out_type is int:
        return self._NBestEncodeAsIds(text, nbest_size, add_bos, add_eos, reverse)
      else:
        return self._NBestEncodeAsPieces(text, nbest_size, add_bos, add_eos, reverse, emit_unk_piece)

    if type(input) is list:
      return [_encode(n) for n in input]

    return _encode(input)


  def SampleEncodeAndScore(self,
                           input,
                           out_type=None,
                           add_bos=None,
                           add_eos=None,
                           reverse=None,
                           emit_unk_piece=None,
                           num_samples=None,
                           theta=None,
                           wor=None,
                           include_best=None):
    """SampleEncodeAndScore text input to segmented ids or tokens.

      Args:
      input: input string. accepsts list of string.
      out_type: output type. int or str.
      add_bos: Add <s> to the result (Default = false)
      add_eos: Add </s> to the result (Default = false) <s>/</s> is added after reversing (if enabled).
      reverse: Reverses the tokenized sequence (Default = false)
      emit_unk_piece: Emits the unk literal string (Default = false)
      num_samples: How many samples to return (Default = 1)
      theta: inverse temperature for sampling
      wor: whether to sample without replacement (Default = false)
      include_best: whether to include the best tokenization, requires wor=True (Default = false)
    """

    if out_type is None:
      out_type = self._out_type
    if add_bos is None:
      add_bos = self._add_bos
    if add_eos is None:
      add_eos = self._add_eos
    if reverse is None:
      reverse = self._reverse
    if emit_unk_piece is None:
      emit_unk_piece = self._emit_unk_piece
    if num_samples is None:
      num_samples = 1
    if theta is None:
      theta = 1.
    if wor is None:
      wor = False
    if include_best is None:
      include_best = False

    if num_samples <= 0:
      raise RuntimeError('num_examples must be positive')

    if include_best and not wor:
      raise RuntimeError('When include_best is True, We must specify "wor = True".')
                        

    def _encode(text):
      if out_type is int:
        return self._SampleEncodeAndScoreAsIds(text, num_samples, theta, wor, include_best,
                                               add_bos, add_eos, reverse)
      else:
        return self._SampleEncodeAndScoreAsPieces(text, num_samples, theta, wor, include_best,
                                                  add_bos, add_eos, reverse, emit_unk_piece)

    if type(input) is list:
      return [_encode(n) for n in input]

    return _encode(input)


  def Decode(self, input):
    """Decode processed id or token sequences."""

    if not input:
      return self.DecodeIds([])
    elif type(input) is int:
      return self.DecodeIdsWithCheck([input])
    elif type(input) is str:
      return self.DecodePieces([input])

    def _decode(input):
      if not input:
        return self.DecodeIds([])
      if type(input[0]) is int:
        return self.DecodeIdsWithCheck(input)
      return self.DecodePieces(input)

    if type(input[0]) is list:
      return [_decode(n) for n in input]

    return _decode(input)


  def Entropy(self, input, theta):
    """Calculate sentence entropy"""

    if type(input) is list:
      return [self.CalculateEntropy(n, theta) for n in input]
    return self.CalculateEntropy(input, theta)


  def piece_size(self):
    return self.GetPieceSize()


  def vocab_size(self):
    return self.GetPieceSize()


  def __getstate__(self):
    return self.serialized_model_proto()


  def __setstate__(self, serialized_model_proto):
    self.__init__()
    self.LoadFromSerializedProto(serialized_model_proto)


  def __len__(self):
    return self.GetPieceSize()


  def __getitem__(self, piece):
    return self.PieceToId(piece)


  def Load(self, model_file=None, model_proto=None):
    """Overwride SentencePieceProcessor.Load to support both model_file and model_proto.

    Args:
      model_file: The sentencepiece model file path.
      model_proto: The sentencepiece model serialized proto. Either `model_file`
        or `model_proto` must be set.
    """
    if model_file and model_proto:
      raise RuntimeError('model_file and model_proto must be exclusive.')
    if model_proto:
      return self.LoadFromSerializedProto(model_proto)
    return self.LoadFromFile(model_file)
}
}

%extend sentencepiece::SentencePieceTrainer {
  static void _TrainFromString(absl::string_view arg) {
    const auto _status = sentencepiece::SentencePieceTrainer::Train(arg);
    if (!_status.ok()) throw _status;
    return;
  }

  static void _TrainFromMap(const std::unordered_map<std::string, std::string> &args) {
    const auto _status = sentencepiece::SentencePieceTrainer::Train(args);
    if (!_status.ok()) throw _status;
    return;
  }

  static void _TrainFromMap2(const std::unordered_map<std::string, std::string> &args,
                            SentenceIterator *iter) {
    const auto _status = sentencepiece::SentencePieceTrainer::Train(args, iter);
    if (!_status.ok()) throw _status;
    return;
  }

  static sentencepiece::util::bytes _TrainFromMap3(const std::unordered_map<std::string, std::string> &args) {
    sentencepiece::util::bytes model_proto;
    const auto _status = sentencepiece::SentencePieceTrainer::Train(args, nullptr, &model_proto);
    if (!_status.ok()) throw _status;
    return model_proto;
  }

  static sentencepiece::util::bytes _TrainFromMap4(const std::unordered_map<std::string, std::string> &args,
                                                  SentenceIterator *iter) {
    sentencepiece::util::bytes model_proto;
    const auto _status = sentencepiece::SentencePieceTrainer::Train(args, iter, &model_proto);
    if (!_status.ok()) throw _status;
    return model_proto;
  }

%pythoncode {
  @staticmethod
  def _Train(arg=None, **kwargs):
    """Train Sentencepiece model. Accept both kwargs and legacy string arg."""
    if arg is not None and type(arg) is str:
      return SentencePieceTrainer._TrainFromString(arg)

    def _encode(value):
      """Encode value to CSV.."""
      if type(value) is list:
        if sys.version_info[0] == 3:
          f = StringIO()
        else:
          f = BytesIO()
        writer = csv.writer(f, lineterminator='')
        writer.writerow([str(v) for v in value])
        return f.getvalue()
      else:
        return str(value)

    sentence_iterator = None
    model_writer = None
    new_kwargs = {}
    for key, value in kwargs.items():
      if key in ['sentence_iterator', 'sentence_reader']:
        sentence_iterator = value
      elif key in ['model_writer']:
        model_writer = value
      else:
        new_kwargs[key] = _encode(value)

    if model_writer:
      if sentence_iterator:
        model_proto = SentencePieceTrainer._TrainFromMap4(new_kwargs,
                                                         sentence_iterator)
      else:
        model_proto = SentencePieceTrainer._TrainFromMap3(new_kwargs)
      model_writer.write(model_proto)
    else:
      if sentence_iterator:
        return SentencePieceTrainer._TrainFromMap2(new_kwargs, sentence_iterator)
      else:
        return SentencePieceTrainer._TrainFromMap(new_kwargs)

    return None

  @staticmethod
  def Train(arg=None, logstream=None, **kwargs):
    with _LogStream(ostream=logstream):
      SentencePieceTrainer._Train(arg=arg, **kwargs)
}
}

%typemap(out) std::vector<int> {
  $result = PyList_New($1.size());
  for (size_t i = 0; i < $1.size(); ++i) {
    PyList_SetItem($result, i, PyInt_FromLong(static_cast<long>($1[i])));
  }
}

%typemap(out) std::vector<std::vector<int>> {
  $result = PyList_New($1.size());
  for (size_t i = 0; i < $1.size(); ++i) {
    PyObject *obj = PyList_New($1[i].size());
    for (size_t j = 0; j < $1[i].size(); ++j) {
      PyList_SetItem(obj, j, PyInt_FromLong(static_cast<long>($1[i][j])));
    }
    PyList_SetItem($result, i, obj);
  }
}

%typemap(out) std::vector<std::string> {
  PyObject *input_type = resultobj;
  $result = PyList_New($1.size());
  for (size_t i = 0; i < $1.size(); ++i) {
    PyList_SetItem($result, i, MakePyOutputString($1[i], input_type));
  }
}

%typemap(out) std::vector<std::vector<std::string>> {
  PyObject *input_type = resultobj;
  $result = PyList_New($1.size());
  for (size_t i = 0; i < $1.size(); ++i) {
    PyObject *obj = PyList_New($1[i].size());
    for (size_t j = 0; j < $1[i].size(); ++j) {
      PyList_SetItem(obj, j, MakePyOutputString($1[i][j], input_type));
    }
    PyList_SetItem($result, i, obj);
  }
}

%typemap(out) std::string {
  PyObject *input_type = resultobj;
  $result = MakePyOutputString($1, input_type);
}

%typemap(out) const std::string& {
  PyObject *input_type = resultobj;
  $result = MakePyOutputString(*$1, input_type);
}

%typemap(out) sentencepiece::util::bytes {
  $result = MakePyOutputBytes($1);
}

%typemap(out) sentencepiece::util::Status {
  if (!$1.ok()) {
    SWIG_exception(ToSwigError($1.code()), $1.ToString().c_str());
  }
  $result = SWIG_From_bool($1.ok());
}

%typemap(in) const std::string & {
  const PyInputString ustring($input);
  if (!ustring.IsAvalable()) {
    PyErr_SetString(PyExc_TypeError, "not a string");
    SWIG_fail;
  }
  resultobj = ustring.input_type();
  $1 = new std::string(ustring.data(), ustring.size());
}

%typemap(typecheck) absl::string_view = char *;

%typemap(in) absl::string_view {
  const PyInputString ustring($input);
  if (!ustring.IsAvalable()) {
    PyErr_SetString(PyExc_TypeError, "not a string");
    SWIG_fail;
  }
  resultobj = ustring.input_type();
  $1 = absl::string_view(ustring.data(), ustring.size());
}

%typemap(in) const std::vector<std::string>& {
  std::vector<std::string> *out = nullptr;
  if (PyList_Check($input)) {
    const size_t size = PyList_Size($input);
    out = new std::vector<std::string>(size);
    for (size_t i = 0; i < size; ++i) {
      const PyInputString ustring(PyList_GetItem($input, i));
      if (ustring.IsAvalable()) {
        (*out)[i] = std::string(ustring.data(), ustring.size());
      } else {
        PyErr_SetString(PyExc_TypeError, "list must contain strings");
        SWIG_fail;
      }
      resultobj = ustring.input_type();
    }
  } else {
    PyErr_SetString(PyExc_TypeError, "not a list");
    SWIG_fail;
  }
  $1 = out;
}

%typemap(in) const std::vector<int>& {
  std::vector<int> *out = nullptr;
  if (PyList_Check($input)) {
    const size_t size = PyList_Size($input);
    out = new std::vector<int>(size);
    for (size_t i = 0; i < size; ++i) {
      PyObject *o = PyList_GetItem($input, i);
      if (PyInt_Check(o)) {
        (*out)[i] = static_cast<int>(PyInt_AsLong(o));
      } else {
        PyErr_SetString(PyExc_TypeError,"list must contain integers");
        SWIG_fail;
      }
    }
  } else {
    PyErr_SetString(PyExc_TypeError,"not a list");
    SWIG_fail;
  }
  $1 = out;
}

%typemap(in) const std::unordered_map<std::string, std::string> & {
  std::unordered_map<std::string, std::string> *out = nullptr;
  if (PyDict_Check($input)) {
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    out = new std::unordered_map<std::string, std::string>;
    while (PyDict_Next($input, &pos, &key, &value)) {
      const PyInputString key_ustring(key);
      const PyInputString value_ustring(value);
      if (key_ustring.IsAvalable() && value_ustring.IsAvalable()) {
        out->emplace(std::string(key_ustring.data(), key_ustring.size()),
                     std::string(value_ustring.data(), value_ustring.size()));
      } else {
        PyErr_SetString(PyExc_TypeError, "map must contain strings.");
        SWIG_fail;
      }
      resultobj = key_ustring.input_type();
    }
  } else {
    PyErr_SetString(PyExc_TypeError, "not a dictionary");
    SWIG_fail;
  }
  $1 = out;
}

%typemap(out) std::vector<std::pair<std::vector<std::string>, float>> {
  PyObject *input_type = resultobj;
  $result = PyList_New($1.size());
  for (size_t i = 0; i < $1.size(); ++i) {
    PyObject *obj = PyList_New($1[i].first.size());
    for (size_t j = 0; j < $1[i].first.size(); ++j) {
      PyList_SetItem(obj, j, MakePyOutputString($1[i].first[j], input_type));
    }
    PyList_SetItem($result, i, PyTuple_Pack(2, obj, PyFloat_FromDouble(static_cast<double>($1[i].second))));
  }
}

%typemap(out) std::vector<std::pair<std::vector<int>, float>> {
  $result = PyList_New($1.size());
  for (size_t i = 0; i < $1.size(); ++i) {
    PyObject *obj = PyList_New($1[i].first.size());
    for (size_t j = 0; j < $1[i].first.size(); ++j) {
      PyList_SetItem(obj, j, PyInt_FromLong(static_cast<long>($1[i].first[j])));
    }
    PyList_SetItem($result, i, PyTuple_Pack(2, obj, PyFloat_FromDouble(static_cast<double>($1[i].second))));
  }
}

%typemap(in) sentencepiece::SentenceIterator * {
  sentencepiece::SentenceIterator *out = nullptr;
  if (PyIter_Check($input)) {
    out = new PySentenceIterator($input);
  } else {
    PyErr_SetString(PyExc_TypeError, "not a iterator");
    SWIG_fail;
  }
  $1 = out;
}

%typemap(freearg) const std::string& {
  delete $1;
}

%typemap(freearg) const std::vector<std::string>& {
  delete $1;
}

%typemap(freearg) const std::vector<std::vector<std::string>>& {
  delete $1;
}

%typemap(freearg) const std::vector<int>& {
  delete $1;
}

%typemap(freearg) const std::vector<std::vector<int>>& {
  delete $1;
}

%typemap(freearg) const std::unordered_map<std::string, std::string> & {
  delete $1;
}

%typemap(freearg) sentencepiece::SentenceIterator * {
  delete $1;
}

%include <sentencepiece_processor.h>
%include <sentencepiece_trainer.h>

%pythoncode %{

import re
import csv
import sys
import os
from io import StringIO
from io import BytesIO


def _add_snake_case(classname):
  """Added snake_cased method from CammelCased method."""

  snake_map = {}
  for k, v in classname.__dict__.items():
    if re.match(r'^[A-Z]+', k):
      snake = re.sub(r'(?<!^)(?=[A-Z])', '_',
                     k).lower().replace('n_best', 'nbest')
      snake_map[snake] = v
  for k, v in snake_map.items():
    setattr(classname, k, v)


def _batchnize(classname, name):
  """Enables batch request for the method classname.name."""
  func = getattr(classname, name, None)
  def _func(v, n):
    if type(n) is int and (n < 0 or n >= v.piece_size()):
      raise IndexError('piece id is out of range.')
    return func(v, n)

  def _batched_func(self, arg):
    if type(arg) is list:
      return [_func(self, n) for n in arg]
    else:
      return _func(self, arg)

  setattr(classname, name, _batched_func)


_sentencepiece_processor_init_native = SentencePieceProcessor.__init__
setattr(SentencePieceProcessor, '__init__', SentencePieceProcessor.Init)

SentencePieceProcessor.Tokenize = SentencePieceProcessor.Encode
SentencePieceProcessor.Detokenize = SentencePieceProcessor.Decode
SentencePieceProcessor.DecodeIds = SentencePieceProcessor.DecodeIdsWithCheck
SentencePieceProcessor.DecodeIdsAsSerializedProto = SentencePieceProcessor.DecodeIdsAsSerializedProtoWithCheck

for m in [
    'PieceToId', 'IdToPiece', 'GetScore', 'IsUnknown', 'IsControl', 'IsUnused',
    'IsByte'
]:
  _batchnize(SentencePieceProcessor, m)

_add_snake_case(SentencePieceProcessor)
_add_snake_case(SentencePieceTrainer)
set_random_generator_seed = SetRandomGeneratorSeed

from ._version import __version__

class _LogStream(object):
  def __init__(self, ostream=None):
    self.ostream = ostream
    if self.ostream is not None:
      self.orig_stream_fileno = sys.stderr.fileno()

  def __enter__(self):
    if self.ostream is not None:
      self.orig_stream_dup = os.dup(self.orig_stream_fileno)
      os.dup2(self.ostream.fileno(), self.orig_stream_fileno)

  def __exit__(self, type, value, traceback):
    if self.ostream is not None:
      os.close(self.orig_stream_fileno)
      os.dup2(self.orig_stream_dup, self.orig_stream_fileno)
      os.close(self.orig_stream_dup)
      self.ostream.close()
%}
