import json
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import numpy 
import triton_python_backend_utils as pb_utils
import time
from models.blip.blip_vqa_text_decoder import blip_vqa_text_decoder

root_path='/dynamic_batch/triton-multi-modal-serving'

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = json.loads(args["model_config"])

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(self.model_config, "OUTPUT0")[
                "data_type"
            ]
        )

        # Instantiate the PyTorch model
        model_url = root_path+"/pretrained/model_base_vqa_capfilt_large.pth"
        self.model = blip_vqa_text_decoder(pretrained=model_url, vit="base")
        self.model.eval()
        self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu")

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -----.--
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        #start=time.time()
        output0_dtype = self.output0_dtype

        responses = []
        in_0s_tmp=[]
        in_1s_tmp=[]
        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        
        for request in requests:
            # Get INPUT0
            in_0s_tmp.append(pb_utils.get_input_tensor_by_name(request, "INPUT0").as_numpy())
            input1_data_shape=(1,in_0s_tmp[-1].shape[2])
            in_1s_tmp.append(torch.ones(input1_data_shape, dtype=torch.long).numpy(force=True))

        max_len=max(in_1.shape[-1] for in_1 in in_1s_tmp)

        in_0s=[]
        in_1s=[]

        for i in range(len(requests)):
            if in_1s_tmp[i].shape[-1]<max_len:
                in_0s.append(np.pad(in_0s_tmp[i],((0,0),(0,0),(max_len-in_1s_tmp[i].shape[-1],0),(0,0)),"constant"))
                in_1s.append(np.pad(in_1s_tmp[i],((0,0),(max_len-in_1s_tmp[i].shape[-1],0)),"constant",constant_values=(0, 0)))
            else:
                in_0s.append(in_0s_tmp[i])
                in_1s.append(in_1s_tmp[i])

        in_0=numpy.concatenate(in_0s,axis=0) if len(in_0s)>1 else in_0s[0]
        in_1=numpy.concatenate(in_1s,axis=0) if len(in_1s)>1 else in_1s[0]

        with torch.no_grad():
            out_0s = self.model(in_0,in_1)

        print()
        print(f"blip_vqa_text_decoder:{len(in_0)}")
            # with profile(
            #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            #     record_shapes=True,
            # ) as prof:
            #     with torch.no_grad():
            #         out_0 = self.model(in_0.as_numpy(), in_1.as_numpy())
            #
            # with open("/workspace/profiletable.txt", "w") as f:
            #     print(prof.key_averages().table(), file=f)
        for i in range(len(in_0)):
            out_tensor_0 = pb_utils.Tensor("OUTPUT0", out_0s[i:i+1].astype(output0_dtype))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0]
            )
            responses.append(inference_response)

        #print(start,time.time())
        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
