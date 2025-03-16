from ldpc import bposd_decoder, bp_decoder
import numpy as np
from .decoder import Decoder


class BPOSD_decoder(Decoder):
    def __init__(self,h,p):
        super().__init__("BPOSD")
        self.h = h
        self.p = p
        self.decoder = bposd_decoder(
            self.h,
            error_rate=p,
            channel_probs=[None],
            max_iter= self.h.shape[1],
            bp_method="ms",
            ms_scaling_factor=0,
            osd_method="osd_e",
            osd_order=7,
        )
    def decode(self,syndrome):
        self.decoder.decode(syndrome)
        return self.decoder.osdw_decoding

class BP_decoder(Decoder):
    def __init__(self,h,p):
        super().__init__("BP")
        self.h = h
        self.p = p
        self.decoder = bp_decoder(
            self.h,
            error_rate=p,
            channel_probs=[None],
            max_iter= self.h.shape[1],
            bp_method="ms",
            ms_scaling_factor=0,
        )
    def decode(self,syndrome):
        self.decoder.decode(syndrome)
        return self.decoder.bp_decoding

