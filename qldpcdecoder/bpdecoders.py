from ldpc import bposd_decoder, bp_decoder
import numpy as np
from .decoder import Decoder


class BPOSD_decoder(Decoder):
    def __init__(self):
        super().__init__("BPOSD")
    def set_h(self,h,prior,p):
        self.h = h
        self.decoder = bposd_decoder(
            self.h,
            error_rate=p,
            channel_probs=prior,
            max_iter= self.h.shape[1],
            bp_method="ms",
            ms_scaling_factor=0,
            osd_method="osd_0",
            osd_order=0,
            input_vector_type="syndrome",
        )
    def decode(self,syndrome):
        self.decoder.decode(syndrome)
        return self.decoder.osdw_decoding

class BP_decoder(Decoder):
    def __init__(self):
        super().__init__("BP")
    def set_h(self,h,prior,p):
        self.h = h
        self.decoder = bp_decoder(
            self.h,
            error_rate=p,
            channel_probs=prior,
            max_iter= self.h.shape[1],
            bp_method="ms",
            ms_scaling_factor=0,
            input_vector_type="syndrome",
            )
    def decode(self,syndrome):
        self.decoder.decode(syndrome)
        return self.decoder.bp_decoding

