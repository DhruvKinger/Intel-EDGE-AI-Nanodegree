#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork,IEPlugin


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.net = None
        self.plugin = None
        self.input_blob = None
        self.out_blob = None
        self.net_plugin = None
        self.infer_request = None
        
    def load_model(self,model,device,cpu_extension=None):
       
        model_xml=model
        model_bin=os.path.splitext(model_xml)[0] + ".bin"
        
        
        ### TODO: Load the model ###
        
        self.plugin = IECore()
        
       if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)
        
        ### TODO: Check for supported layers ###
        ### TODO: Add any necessary extensions ###
        
        
        self.net = IENetwork(model=model_xml, weights=model_bin)
        
        if self.plugin.device=="CPU":
            supported_layers = self.plugin.get_supported_layers(self.net)
        
        unsupported_layers = [layers for layers in self.net.layers.keys() if layers not in supported_layers]

        if len(unsupported_layers) > 0:
            sys.exit(1)

        self.net_plugin = self.plugin.load_network(self.net, device)
       
        self.input_blob = next(iter(self.net.inputs))
        self.out_blob = next(iter(self.net.outputs))
            
            ### TODO: Return the loaded inference plugin ###
        
        
        ### Note: You may need to update the function parameters. ###
        return self.net_plugin,self.get_input_shape()

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        return self.net.inputs[self.input_blob].shape
    
    ###def performance_counter(self, request_id):
       ## perf_count = self.net_plugin.requests[request_id].get_perf_counts()
        
        ##return perf_count

    def exec_net(self,request_id,frame):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        self.infer_request = self.net_plugin.start_async(
        request_id=request_id, inputs={self.input_blob: frame})
        
        return self.net_plugin

    def wait(self,request_id):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        waiting = self.net_plugin.requests[request_id].wait(-1)
        return waiting

    def get_output(self,request_id, output=None):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        if output:
            res = self.infer_request_handle.outputs[output]
        else:
            res = self.net_plugin.requests[request_id].outputs[self.out_blob]
       
        return res

    def clean(self):
        del self.net_plugin
        del self.plugin
        del self.net   
