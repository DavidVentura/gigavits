import struct
def varint(b,i):
    r=s=0
    while True:
        x=b[i]; i+=1; r|=(x&0x7f)<<s; s+=7
        if not x&0x80: return r,i
def decode_prefix(b, i, want):
    n,i=varint(b,i); out=bytearray()
    while len(out)<want and i<len(b):
        tag=b[i]; i+=1; t=tag&3
        if t==0:
            l=tag>>2
            if l>=60:
                nb=l-59; l=int.from_bytes(b[i:i+nb],'little'); i+=nb
            l+=1; out+=b[i:i+l]; i+=l
        else:
            if t==1: l=((tag>>2)&7)+4; off=((tag>>5)<<8)|b[i]; i+=1
            elif t==2: l=(tag>>2)+1; off=int.from_bytes(b[i:i+2],'little'); i+=2
            else: l=(tag>>2)+1; off=int.from_bytes(b[i:i+4],'little'); i+=4
            for _ in range(l): out.append(out[-off])
    return n,bytes(out)
