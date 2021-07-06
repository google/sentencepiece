%module sentencepiece
%include exception.i

%{
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
#if PY_VERSION_HEX >= 0x03000000
    if (PyUnicode_Check(obj)) {
       // Python3, Unicode
      str_ = const_cast<char *>(PyUnicode_AsUTF8AndSize(obj, &size_));
      input_type_ = kUnicodeInput;
    } else if (PyBytes_Check(obj)) {
       // Python3, Bytes
      PyBytes_AsStringAndSize(obj, &str_, &size_);
      input_type_ = kByteInput;
    }
#else
    if (PyUnicode_Check(obj)) {
      // Python2, Unicode
      PyObject *utf8_obj = PyUnicode_AsUTF8String(obj);
      PyString_AsStringAndSize(utf8_obj, &str_, &size_);
      input_type_ = utf8_obj;
    } else if (PyString_Check(obj)) {
      // Python2, Bytes,
      PyString_AsStringAndSize(obj, &str_, &size_);
      input_type_ = kByteInput;
    }
#endif
    else {
      str_ = nullptr;
    }
  }
  const char* data() const { return str_; }
  Py_ssize_t size() const { return size_; }
  bool IsAvalable() const { return str_ != nullptr; }
  PyObject *input_type() const { return input_type_; }

  static bool IsUnicode(PyObject *resultobj) {
#if PY_VERSION_HEX >= 0x03000000
    return (resultobj == nullptr || resultobj == kUnicodeInput);
#else
    return (resultobj != nullptr && resultobj != kByteInput);
#endif
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
#if PY_VERSION_HEX >= 0x03000000
  return PyBytes_FromStringAndSize(output.data(), output.size());
#else
  return PyString_FromStringAndSize(output.data(), output.size());
#endif
}

PyObject* MakePyOutputBytes(const std::string& output) {
#if PY_VERSION_HEX >= 0x03000000
  return PyBytes_FromStringAndSize(output.data(), output.size());
#else
  return PyString_FromStringAndSize(output.data(), output.size());
#endif
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
}
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
    for (int id : ids)
      if (id < 0 || id >= num_pieces)
        throw sentencepiece::util::Status(
            sentencepiece::util::StatusCode::kOutOfRange,
            "piece id is out of range.");
    return $self->DecodeIds(ids);
  }

  util::bytes DecodeIdsAsSerializedProtoWithCheck(
      const std::vector<int> &ids) const {
    const int num_pieces = $self->GetPieceSize(); 
    for (int id : ids)
      if (id < 0 || id >= num_pieces)
        throw sentencepiece::util::Status(
            sentencepiece::util::StatusCode::kOutOfRange,
            "piece id is out of range.");
    return $self->DecodeIdsAsSerializedProto(ids);
  }

%pythoncode {
  def Init(self,
           model_file=None,
           model_proto=None,
           out_type=int,
           add_bos=False,
           add_eos=False,
           reverse=False,
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
        if enable_sampling:
          result = self.SampleEncodeAsIds(text, nbest_size, alpha)
        else:
          result = self.EncodeAsIds(text)
      else:
        if enable_sampling:
          result = self.SampleEncodeAsPieces(text, nbest_size, alpha)
        else:
          result = self.EncodeAsPieces(text)

      if reverse:
        result.reverse()
      if add_bos:
        if out_type is int:
          result = [self.bos_id()] + result
        else:
          result = [self.IdToPiece(self.bos_id())] + result

      if add_eos:
        if out_type is int:
          result = result + [self.eos_id()]
        else:
          result = result + [self.IdToPiece(self.eos_id())]

      return result

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
  def Train(arg=None, **kwargs):
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
%}
