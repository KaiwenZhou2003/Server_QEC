from .gauss_decoder import guass_decoder
from .decoder import Decoder
from .basis_compute import compute_basis_complement
import numpy as np
import os
import subprocess

class HomologyDecoder(Decoder):
    def __init__(self, code,p, **kwargs):
        super().__init__("HomologyDecoder")
        self.code = code
        self.p = p
        self.guassdecoder = guass_decoder(p=p)
        self.guassdecoder.set_h(code.hx,None,p)
    

    def get_init_error(self, syndrome):
        """ 计算初始错误分布 """
        return self.guassdecoder.decode(syndrome,order=0)
        
        
    def decode(self, syndrome):
        # return self.exhaustive_decode(syndrome)
        return self.homology_decode(syndrome)
    
    def homology_decode(self, syndrome):
        """ 基于同余类解码 """
        

                
   
   
        
            