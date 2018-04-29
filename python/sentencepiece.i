%module sentencepiece
%include exception.i

%{
#include <sentencepiece_processor.h>
#include <sentencepiece_trainer.h>

namespace {
PyObject* kStringInput = reinterpret_cast<PyObject* >(0x1);
PyObject* kUnicodeInput = reinterpret_cast<PyObject* >(0x2);

class PyInputString {
 public:
  explicit PyInputString(PyObject* obj) {
#if PY_VERSION_HEX >= 0x03000000
    if (PyUnicode_Check(obj)) {
      str_ = PyUnicode_AsUTF8AndSize(obj, &size_);
      input_type_ = kUnicodeInput;
    } else if (PyBytes_Check(obj)) {
      PyBytes_AsStringAndSize(obj, &str_, &size_);
      input_type_ = kStringInput;
    }
#else
    if (PyUnicode_Check(obj)) {
      utf8_obj_ = PyUnicode_AsUTF8String(obj);
      PyString_AsStringAndSize(utf8_obj_, &str_, &size_);
      input_type_ = kUnicodeInput;
    } else if (PyString_Check(obj)) {
      PyString_AsStringAndSize(obj, &str_, &size_);
      input_type_ = kStringInput;
    }
#endif
    else {
      str_ = nullptr;
    }
  }
  virtual ~PyInputString() {
    Py_XDECREF(utf8_obj_);
  }
  const char* str() const { return str_; }
  Py_ssize_t size() const { return size_; }
  bool IsAvalable() const { return str_ != nullptr; }
  PyObject *input_type() const { return input_type_; }

 private:
  PyObject* utf8_obj_ = nullptr;
  PyObject* input_type_ = nullptr;
  char* str_ = nullptr;
  Py_ssize_t size_ = 0;
};

PyObject* MakePyOutputString(const std::string& output, PyObject *resultobj) {
#if PY_VERSION_HEX >= 0x03000000
  return resultobj == kStringInput ?
      PyBytes_FromStringAndSize(output.data(), output.size()) :
      PyUnicode_FromStringAndSize(output.data(), output.size());
#else
   return resultobj == kUnicodeInput ?
       PyUnicode_FromStringAndSize(output.data(), output.size()) :
       PyString_FromStringAndSize(output.data(), output.size());
#endif
}

int ToSwigError(sentencepiece::util::error::Code code) {
  switch (code) {
    case sentencepiece::util::error::NOT_FOUND:
      return SWIG_IOError;
    case sentencepiece::util::error::OUT_OF_RANGE:
      return SWIG_IndexError;
    case sentencepiece::util::error::INVALID_ARGUMENT:
      return SWIG_SyntaxError;
    default:
      return SWIG_RuntimeError;
  }
  return SWIG_RuntimeError;
}

#define THROW_IF_ERROR(expr)                            \
  do {                                                  \
    const sentencepiece::util::Status _status = expr;   \
    if (!_status.ok()) throw _status;                   \
  } while (0)

}  // namespace
%}

%exception {
  try { $action }
  catch (const sentencepiece::util::Status &status) {
    SWIG_exception(ToSwigError(status.code()), status.ToString().c_str());
  }
}

%ignore sentencepiece::util::Status;
%ignore sentencepiece::util::error::Code;

%ignore sentencepiece::SentencePieceProcessor::status() const;
%ignore sentencepiece::SentencePieceProcessor::Encode(std::string const &, std::vector<std::string>*) const;
%ignore sentencepiece::SentencePieceProcessor::Encode(std::string const &, std::vector<int>*) const;
%ignore sentencepiece::SentencePieceProcessor::Encode(std::string const &, SentencePieceText *) const;
%ignore sentencepiece::SentencePieceProcessor::SampleEncode(std::string const &,int,float, std::vector< std::string > *) const;
%ignore sentencepiece::SentencePieceProcessor::SampleEncode(std::string const &,int,float, std::vector< int > *) const;
%ignore sentencepiece::SentencePieceProcessor::SampleEncode(std::string const &,int,float, SentencePieceText *) const;
%ignore sentencepiece::SentencePieceProcessor::NBestEncode(std::string const &,int, NBestSentencePieceText *) const;
%ignore sentencepiece::SentencePieceProcessor::NBestEncode(std::string const &,int,std::vector< std::vector< std::string > > *) const;
%ignore sentencepiece::SentencePieceProcessor::NBestEncode(std::string const &,int,std::vector< std::vector< int > > *) const;
%ignore sentencepiece::SentencePieceProcessor::Decode(std::vector<std::string> const &, std::string *) const;
%ignore sentencepiece::SentencePieceProcessor::Decode(std::vector<int> const &, std::string *) const;
%ignore sentencepiece::SentencePieceProcessor::Decode(std::vector<std::string> const &, SentencePieceText *) const;
%ignore sentencepiece::SentencePieceProcessor::Decode(std::vector<int> const &, SentencePieceText *) const;
%ignore sentencepiece::SentencePieceProcessor::model_proto;
%ignore sentencepiece::SentencePieceProcessor::Load(std::istream *);
%ignore sentencepiece::SentencePieceProcessor::LoadOrDie(std::istream *);
%ignore sentencepiece::SentencePieceProcessor::model_proto();
%ignore sentencepiece::SentencePieceTrainer::Train(int, char **);
%ignore sentencepiece::SentencePieceTrainer::Train(const TrainerSpec &);
%ignore sentencepiece::SentencePieceTrainer::Train(const TrainerSpec &, const NormalizerSpec &);

%extend sentencepiece::SentencePieceProcessor {
  std::vector<std::string> Encode(const std::string& input) const {
    std::vector<std::string> output;
    THROW_IF_ERROR($self->Encode(input, &output));
    return output;
  }

  std::vector<std::string> EncodeAsPieces(const std::string& input) const {
    std::vector<std::string> output;
    THROW_IF_ERROR($self->Encode(input, &output));
    return output;
  }

  std::vector<int> EncodeAsIds(const std::string& input) const {
    std::vector<int> output;
    THROW_IF_ERROR($self->Encode(input, &output));
    return output;
  }

  std::vector<std::vector<std::string>> NBestEncode(const std::string& input, int nbest_size) const {
    std::vector<std::vector<std::string>> output;
    THROW_IF_ERROR($self->NBestEncode(input, nbest_size, &output));
    return output;
  }

  std::vector<std::vector<std::string>> NBestEncodeAsPieces(const std::string& input, int nbest_size) const {
    std::vector<std::vector<std::string>> output;
    THROW_IF_ERROR($self->NBestEncode(input, nbest_size, &output));
    return output;
  }

  std::vector<std::vector<int>> NBestEncodeAsIds(const std::string& input, int nbest_size) const {
    std::vector<std::vector<int>> output;
    THROW_IF_ERROR($self->NBestEncode(input, nbest_size, &output));
    return output;
  }

  std::vector<std::string> SampleEncode(const std::string& input, int nbest_size, float alpha) const {
    std::vector<std::string> output;
    THROW_IF_ERROR($self->SampleEncode(input, nbest_size, alpha, &output));
    return output;
  }

  std::vector<std::string> SampleEncodeAsPieces(const std::string& input, int nbest_size, float alpha) const {
    std::vector<std::string> output;
    THROW_IF_ERROR($self->SampleEncode(input, nbest_size, alpha, &output));
    return output;
  }

  std::vector<int> SampleEncodeAsIds(const std::string& input, int nbest_size, float alpha) const {
    std::vector<int> output;
    THROW_IF_ERROR($self->SampleEncode(input, nbest_size, alpha, &output));
    return output;
  }

  std::string Decode(const std::vector<std::string>& input) const {
    std::string output;
    THROW_IF_ERROR($self->Decode(input, &output));
    return output;
  }

  std::string DecodePieces(const std::vector<std::string>& input) const {
    std::string output;
    THROW_IF_ERROR($self->Decode(input, &output));
    return output;
  }

  std::string DecodeIds(const std::vector<int>& input) const {
    std::string output;
    THROW_IF_ERROR($self->Decode(input, &output));
    return output;
  }

  int __len__() {
    return $self->GetPieceSize();
  }

  int __getitem__(const std::string& key) const {
    return $self->PieceToId(key);
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
    return nullptr;
  }
  resultobj = ustring.input_type();
  $1 = new std::string(ustring.str(), ustring.size());
}

%typemap(in) const std::vector<std::string>& {
  std::vector<std::string> *out = nullptr;
  if (PyList_Check($input)) {
    const size_t size = PyList_Size($input);
    out = new std::vector<std::string>(size);
    for (size_t i = 0; i < size; ++i) {
      const PyInputString ustring(PyList_GetItem($input, i));
      if (ustring.IsAvalable()) {
        (*out)[i] = std::string(ustring.str(), ustring.size());
      } else {
        PyErr_SetString(PyExc_TypeError, "list must contain strings");
        return nullptr;
      }
      resultobj = ustring.input_type();
    }
  } else {
    PyErr_SetString(PyExc_TypeError, "not a list");
    return nullptr;
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
        return nullptr;
      }
    }
  } else {
    PyErr_SetString(PyExc_TypeError,"not a list");
    return nullptr;
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

%include <sentencepiece_processor.h>
%include <sentencepiece_trainer.h>
