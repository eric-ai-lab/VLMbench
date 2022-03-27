import torch
import numpy as np
import torch.nn.functional as F

import cliport.models as models
import cliport.models.core.fusion as fusion
from cliport.models.core.transport import Transport, Transport6Dof
from language_tasks.baseline_models import mlp
from time import time

class TwoStreamTransportLangFusion(Transport6Dof):
    """Two Stream Transport (a.k.a Place) module"""

    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device,  z_roll_pitch=False, joint_all=False):
        self.fusion_type = cfg['train']['trans_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device,  z_roll_pitch=z_roll_pitch, joint_all=joint_all)
    def _build_nets(self):
        stream_one_fcn, stream_two_fcn = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]
        stream_two_model = models.names[stream_two_fcn]

        self.key_stream_one = stream_one_model(self.in_shape, self.output_dim, self.cfg, self.device, self.preprocess)
        self.key_stream_two = stream_two_model(self.in_shape, self.output_dim, self.cfg, self.device, self.preprocess)
        self.query_stream_one = stream_one_model(self.kernel_shape, self.kernel_dim, self.cfg, self.device, self.preprocess)
        self.query_stream_two = stream_two_model(self.kernel_shape, self.kernel_dim, self.cfg, self.device, self.preprocess)
        self.fusion_key = fusion.names[self.fusion_type](input_dim=self.kernel_dim)
        self.fusion_query = fusion.names[self.fusion_type](input_dim=self.kernel_dim)

        print(f"Transport FCN - Stream One: {stream_one_fcn}, Stream Two: {stream_two_fcn}, Stream Fusion: {self.fusion_type}")

    def transport(self, in_tensor, crop, l):
        logits = self.fusion_key(self.key_stream_one(in_tensor), self.key_stream_two(in_tensor, l))
        kernel = self.fusion_query(self.query_stream_one(crop), self.query_stream_two(crop, l))
        return logits, kernel

    def forward(self, inp_img, p, lang_goal, softmax=True):
        """Forward pass."""
        padding = np.zeros((4,2),dtype=int)
        padding[1:,:] = self.padding
        img_unprocessed = np.pad(inp_img, padding, mode='constant')
        input_data = img_unprocessed
        # in_shape = (1,) + input_data.shape
        # input_data = input_data.reshape(in_shape)
        in_tensor = torch.from_numpy(input_data).to(dtype=torch.float, device=self.device)

        # Rotation pivot.
        # pv = np.array([p[0], p[1]]) + self.pad_size
        pv = p + self.pad_size

        # Crop before network (default for Transporters CoRL 2020).
        hcrop = self.pad_size
        in_tensor = in_tensor.permute(0, 3, 1, 2).contiguous()
        b, c, w, h = in_tensor.shape
        crop = in_tensor.unsqueeze(1).repeat(1, self.n_rotations, 1, 1, 1)
        crop = self.rotator(crop, pivot=np.flip(pv,axis=1).copy(), reverse=True)
        crop = torch.stack(crop, dim=1)
        # logits, kernels = [], []
        # for i in range(b):
        #     c = crop[i, :, :, pv[i,0]-hcrop:pv[i,0]+hcrop, pv[i,1]-hcrop:pv[i,1]+hcrop]
        #     """
        #     import cv2
        #     img = c[0,:3].permute(1,2,0).detach().cpu().numpy()
        #     img2 = crop[i, 0, :3].permute(1,2,0).detach().cpu().numpy()
        #     window_name = lang_goal[i]
        #     window_name_crop = lang_goal[i]+"_crop"
        #     # Using cv2.imshow() method 
        #     # Displaying the image 
        #     cv2.imshow(window_name_crop, cv2.cvtColor(np.uint8(img),cv2.COLOR_RGB2BGR))
        #     cv2.imshow(window_name, cv2.cvtColor(np.uint8(img2),cv2.COLOR_RGB2BGR))
            
        #     #waits for user to press any key 
        #     #(this is necessary to avoid Python kernel form crashing)
        #     cv2.waitKey(0) 
            
        #     #closing all open windows 
        #     cv2.destroyAllWindows()
        #     """
        #     logit, kernel = self.transport(in_tensor[i:i+1], c, [lang_goal[i]])
        #     logits.append(logit)
        #     kernels.append(kernel)
        crop = [crop[i, :, :, pv[i,0]-hcrop:pv[i,0]+hcrop, pv[i,1]-hcrop:pv[i,1]+hcrop] for i in range(crop.shape[0])]
        crop = torch.cat(crop, dim=0)
        logits, kernels = self.transport(in_tensor, crop, lang_goal)
        kernels = kernels.reshape(torch.Size([-1, self.n_rotations])+kernels.shape[1:])
        # TODO(Mohit): Crop after network. Broken for now.
        # # Crop after network (for receptive field, and more elegant).
        # in_tensor = in_tensor.permute(0, 3, 1, 2)
        # logits, crop = self.transport(in_tensor, lang_goal)
        # crop = crop.repeat(self.n_rotations, 1, 1, 1)
        # crop = self.rotator(crop, pivot=pv)
        # crop = torch.cat(crop, dim=0)
        # hcrop = self.pad_size
        # kernel = crop[:, :, pv[0]-hcrop:pv[0]+hcrop, pv[1]-hcrop:pv[1]+hcrop]
        # logits = torch.cat(logits,dim=0).contiguous() # B,C,W,H
        # kernels = torch.stack(kernels,dim=0).contiguous() # B,N_rotation,C,w_k,h_k
        return self.correlate(logits, kernels, softmax)


class TwoStreamTransportLangFusionLat(TwoStreamTransportLangFusion):
    """Two Stream Transport (a.k.a Place) module with lateral connections"""

    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device, z_roll_pitch=False, joint_all=False):

        self.fusion_type = cfg['train']['trans_stream_fusion_type']
        # self.output_dim = 3 if not z_roll_pitch else 16
        # self.kernel_dim = 3 if not z_roll_pitch else 16
        # self.z_roll_pitch = z_roll_pitch
        super().__init__(stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device,  z_roll_pitch=z_roll_pitch, joint_all=joint_all)
        # if z_roll_pitch:
        #     self.z_regressor = mlp(input_dim=1, hidden_dim=32, output_dim=1, hidden_depth=3, output_mod=None, device=device)
        #     self.roll_regressor = mlp(input_dim=1, hidden_dim=32, output_dim=n_rotations, hidden_depth=3, output_mod=None, device=device)
        #     self.pitch_regressor = mlp(input_dim=1, hidden_dim=32, output_dim=n_rotations, hidden_depth=3, output_mod=None, device=device)
    def transport(self, in_tensor, crop, l):
        key_out_one, key_lat_one = self.key_stream_one(in_tensor)
        key_out_two = self.key_stream_two(in_tensor, key_lat_one, l)
        logits = self.fusion_key(key_out_one, key_out_two)

        query_out_one, query_lat_one = self.query_stream_one(crop)
        query_out_two = self.query_stream_two(crop, query_lat_one, l)
        kernel = self.fusion_query(query_out_one, query_out_two)

        return logits, kernel
    
    # def correlate(self, in0, in1, softmax):
    #     """Correlate two input tensors."""
    #     # b, c, _, _ = in0.shape
    #     # in1 = in1.reshape(b, self.n_rotations, c, in1.shape[-2], in1.shape[-1])
    #     assert in0.shape[0] == in1.shape[0]
    #     outputs = []
    #     z_tensors, roll_tensors, pitch_tensors = [],[],[]
    #     if not self.z_roll_pitch:
    #         for i in range(in0.shape[0]):
    #             output = F.conv2d(in0[i:i+1], in1[i], padding=(self.pad_size, self.pad_size))
    #             output = F.interpolate(output, size=(in0.shape[-2], in0.shape[-1]), mode='bilinear')
    #             # output = in0[i:i+1]
    #             output = output[:,:,self.pad_size:-self.pad_size, self.pad_size:-self.pad_size]
    #             outputs.append(output)
    #         outputs = torch.cat(outputs, dim=0)
    #         if softmax:
    #             outputs_shape = outputs.shape
    #             outputs = outputs.reshape((1, np.prod(outputs.shape)))
    #             outputs = F.softmax(outputs, dim=-1)
    #             outputs = outputs.reshape(outputs_shape[1:])
    #     else:
    #         for i in range(in0.shape[0]):
    #             output = F.conv2d(in0[i:i+1, :4], in1[i, :, :4], padding=(self.pad_size, self.pad_size))
    #             output = F.interpolate(output, size=(in0.shape[-2], in0.shape[-1]), mode='bilinear')
    #             # output = in0[i:i+1]
    #             output = output[:,:,self.pad_size:-self.pad_size, self.pad_size:-self.pad_size]
    #             outputs.append(output)

    #             z_tensor = F.conv2d(in0[i:i+1, 4:8], in1[i, :, 4:8], padding=(self.pad_size, self.pad_size))
    #             z_tensor = F.interpolate(z_tensor, size=(in0.shape[-2], in0.shape[-1]), mode='bilinear')
    #             z_tensor = z_tensor[:,:,self.pad_size:-self.pad_size, self.pad_size:-self.pad_size]
    #             z_tensors.append(z_tensor)

    #             roll_tensor = F.conv2d(in0[i:i+1, 8:12], in1[i, :, 8:12], padding=(self.pad_size, self.pad_size))
    #             roll_tensor = F.interpolate(roll_tensor, size=(in0.shape[-2], in0.shape[-1]), mode='bilinear')
    #             roll_tensor = roll_tensor[:,:,self.pad_size:-self.pad_size, self.pad_size:-self.pad_size]
    #             roll_tensors.append(roll_tensor)

    #             pitch_tensor = F.conv2d(in0[i:i+1, 12:16], in1[i, :, 12:16], padding=(self.pad_size, self.pad_size))
    #             pitch_tensor = F.interpolate(pitch_tensor, size=(in0.shape[-2], in0.shape[-1]), mode='bilinear')
    #             pitch_tensor = pitch_tensor[:,:,self.pad_size:-self.pad_size, self.pad_size:-self.pad_size]
    #             pitch_tensors.append(pitch_tensor)
    #         z_tensors = torch.cat(z_tensors, dim=0)
    #         outputs = torch.cat(outputs, dim=0)
    #         roll_tensors = torch.cat(roll_tensors, dim=0)
    #         pitch_tensors = torch.cat(pitch_tensors, dim=0)
    #         if softmax:
    #             outputs_shape = outputs.shape
    #             outputs = outputs.reshape((1, np.prod(outputs.shape)))
    #             outputs = F.softmax(outputs, dim=-1)
    #             outputs = outputs.reshape(outputs_shape[1:])
    #             z_tensors = z_tensors[0]
    #     return outputs, z_tensors, roll_tensors, pitch_tensors

    # def correlate(self, in0, in1, softmax):
    #     """Correlate two input tensors."""
    #     # b, c, _, _ = in0.shape
    #     # in1 = in1.reshape(b, self.n_rotations, c, in1.shape[-2], in1.shape[-1])
    #     outputs = []
    #     z_tensors, roll_tensors, pitch_tensors = [],[],[]
    #     for i in range(in0.shape[0]):
    #         output = F.conv2d(in0[i:i+1, :3], in1[i, :, :3], padding=(self.pad_size, self.pad_size))
    #         output = F.interpolate(output, size=(in0.shape[-2], in0.shape[-1]), mode='bilinear')
    #         output = output[:,:,self.pad_size:-self.pad_size, self.pad_size:-self.pad_size]
    #         outputs.append(output)

    #         z_tensor = F.conv2d(in0[i:i+1, 3:7], in1[i, :, 3:7], padding=(self.pad_size, self.pad_size))
    #         z_tensor = F.interpolate(z_tensor, size=(in0.shape[-2], in0.shape[-1]), mode='bilinear')
    #         z_tensor = z_tensor[:,:,self.pad_size:-self.pad_size, self.pad_size:-self.pad_size]
    #         z_tensors.append(z_tensor)

    #         roll_tensor = F.conv2d(in0[i:i+1, 7:11], in1[i, :, 7:11], padding=(self.pad_size, self.pad_size))
    #         roll_tensor = F.interpolate(roll_tensor, size=(in0.shape[-2], in0.shape[-1]), mode='bilinear')
    #         roll_tensor = roll_tensor[:,:,self.pad_size:-self.pad_size, self.pad_size:-self.pad_size]
    #         roll_tensors.append(roll_tensor)

    #         pitch_tensor = F.conv2d(in0[i:i+1, 11:15], in1[i, :, 11:15], padding=(self.pad_size, self.pad_size))
    #         pitch_tensor = F.interpolate(pitch_tensor, size=(in0.shape[-2], in0.shape[-1]), mode='bilinear')
    #         pitch_tensor = pitch_tensor[:,:,self.pad_size:-self.pad_size, self.pad_size:-self.pad_size]
    #         pitch_tensors.append(pitch_tensor)

    #     outputs = torch.cat(outputs, dim=0)
    #     z_tensors = torch.cat(z_tensors, dim=0)
    #     roll_tensors = torch.cat(roll_tensors, dim=0)
    #     pitch_tensors = torch.cat(pitch_tensors, dim=0)
    #     if softmax:
    #         outputs_shape = outputs.shape
    #         outputs = outputs.reshape((1, np.prod(outputs.shape)))
    #         outputs = F.softmax(outputs, dim=-1)
    #         outputs = outputs.reshape(outputs_shape[1:])
    #         z_tensors = z_tensors[0]
    #         roll_tensors = roll_tensors[0]
    #         pitch_tensors = pitch_tensors[0]
    #     return outputs, z_tensors, roll_tensors, pitch_tensors
