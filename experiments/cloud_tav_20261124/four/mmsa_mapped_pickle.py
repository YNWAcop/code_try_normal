"""Read trusted legacy numpy pickle payloads through read-only file mappings.

Keeps original bytes/dtypes/shapes and avoids allocating all MOSEI splits at once.
Only supports the ndarray state layout already used by this project's loader.
"""
import io,pickle,struct
from pathlib import Path
import numpy as np

class Payload:
    def __init__(self,path,offset,size):self.path,self.offset,self.size=path,offset,size

class ArrayProxy:
    def __init__(self,*args):self.array=None
    def __setstate__(self,state):
        version,shape,dtype,fortran,data=state
        if isinstance(data,Payload):
            assert not dtype.hasobject and int(np.prod(shape))*dtype.itemsize==data.size
            self.array=np.memmap(data.path,mode='r',offset=data.offset,dtype=dtype,shape=shape,order='F' if fortran else 'C')
        elif isinstance(data,bytes) and not dtype.hasobject:
            self.array=np.frombuffer(data,dtype=dtype).reshape(shape,order='F' if fortran else 'C')
        else:
            arr=np.empty(0,dtype=np.uint8);arr.__setstate__(state);self.array=arr

class MappedUnpickler(pickle._Unpickler):
    dispatch=pickle._Unpickler.dispatch.copy()
    def __init__(self,file):
        super().__init__(file);self.source=file;self.path=Path(file.name).resolve()
    def find_class(self,module,name):
        if module in ('numpy.core.multiarray','numpy._core.multiarray') and name=='_reconstruct':return ArrayProxy
        return super().find_class(module,name)
    def payload(self,n):
        shape=self.stack[-3] if len(self.stack)>=3 else None
        dtype=self.stack[-2] if len(self.stack)>=2 else None
        frame=self._unframer.current_frame
        remaining=0 if frame is None else frame.getbuffer().nbytes-frame.tell()
        if n>=1024**2 and isinstance(shape,tuple) and isinstance(dtype,np.dtype) and not dtype.hasobject and remaining==0:
            offset=self.source.tell()
            self.source.seek(n,io.SEEK_CUR)
            if self.source.tell()>self.path.stat().st_size:raise EOFError('Truncated array payload')
            self._unframer.current_frame=None
            self.append(Payload(self.path,offset,n))
        else:self.append(self.read(n))
    def load_binbytes(self):self.payload(struct.unpack('<I',self.read(4))[0])
    def load_binbytes8(self):self.payload(struct.unpack('<Q',self.read(8))[0])
    dispatch[pickle.BINBYTES[0]]=load_binbytes
    dispatch[pickle.BINBYTES8[0]]=load_binbytes8

def load_fields(path,fields,splits=('train','valid','test'),audit_test_ids=False):
    with open(path,'rb') as f:raw=MappedUnpickler(f).load()
    def unwrap(v):return v.array if isinstance(v,ArrayProxy) else v
    result={s:{k:unwrap(raw[s][k]) for k in fields} for s in splits}
    if audit_test_ids:result['_audit_test_ids']=unwrap(raw['test']['id'])
    return result
