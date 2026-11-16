"""Accelerated diagnostic profile: unchanged sampled pixels, compact evidence.

Runs only in sbatch. Old batches remain immutable; this profile has its own version,
source hash, timings, and output directory. It does not claim calibrated validity.
"""
import hashlib
from pathlib import Path
import subprocess
import time

from . import qwen_v3 as q
from .io import sha256

VERSION = 'qwen-v3.5-cached-raw-native-frames-gpu-preprocessing-compact-diagnostic-v1'
OriginalNativeQwen = q.NativeQwen
original_prompt_for = q.prompt_for


def compact_prompt(text, task, rubrics):
    return original_prompt_for(text, task, rubrics) + (
        '\nOUTPUT COMPRESSION: use EXACTLY one observation per requested metric, no auxiliary observations. '
        'Keep each detail at most25 words. For motion, put two actually seen consecutive-frame transitions '
        'with their times in that one brief observation. Keep the SAME anchors and grade independently of brevity. '
        'Use all required checks, numeric times, and the requested scores; no other prose. '
        'If background needs an NA reason, keep it under20 words. Return the complete JSON within500 tokens.')


class NativeFrameCache:
    """One source/variant at a time; no PNG scratch files or repeated native decoding."""
    def __init__(self):
        self.key = None
        self.frames = {}

    def get(self, record, task, variant):
        from PIL import Image
        import numpy as np
        started = time.perf_counter()
        key = (record['source_sha256'], variant)
        hit = key == self.key
        if not hit:
            self.frames.clear()
            self.key = None
            selected = sorted({index for view in q.tasks_for(record['info'], record['target_frames'])
                for index in q.content_indices(view, record['info']['frames'], variant)})
            expression = '+'.join(f'eq(n\\,{i})' for i in selected)
            result = subprocess.run([q.binary('ffmpeg'), '-v', 'error', '-threads', '2',
                '-i', record['source_video'], '-vf', f'select={expression}', '-vsync', '0',
                '-frames:v', str(len(selected)), '-threads', '2', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            h,w = record['info']['height'],record['info']['width']
            if len(result.stdout) != len(selected)*h*w*3:
                raise ValueError('Native raw decoder changed frame count or dimensions')
            array = np.frombuffer(result.stdout, dtype=np.uint8).reshape(len(selected),h,w,3)
            self.frames = {index:Image.fromarray(pic).copy() for index,pic in zip(selected,array)}
            self.key = key
        content = q.content_indices(task,record['info']['frames'],variant)
        return [self.frames[i] for i in content],content,{'cache_hit':hit,
            'decode_seconds':time.perf_counter()-started,'cached_native_frames':len(self.frames)}


def image_messages(record, task, ordered, prompt, previous=None):
    """Preserve all native sample/crop pixels and labels from the v3.4 profile."""
    from PIL import Image
    from qwen_vl_utils import smart_resize
    with Image.open(record['image']) as pic:
        reference=pic.convert('RGB').copy()
    images,message,labels=[],[],[]
    def add(pic,label,budget):
        h,w=smart_resize(pic.height,pic.width,factor=32,min_pixels=4096,max_pixels=budget)
        images.append(pic.resize((w,h),Image.Resampling.BICUBIC));labels.append(label)
        message.extend([{'type':'text','text':label},{'type':'image'}])
    add(reference,'CONDITIONING REFERENCE',q.CONFIG['reference_max_pixels'])
    for pic,timestamp in zip(ordered,task['timestamps']):
        add(pic,f'VIDEO FRAME time={timestamp:.6f}s',task['max_pixels'])
    if task['request_id']!='overall_uniform':
        for name,pic in q.quadrants(reference):add(pic,'REFERENCE DETAIL '+name,q.CONFIG['detail_max_pixels'])
        for pos in (0,len(ordered)//2,len(ordered)-1):
            for name,pic in q.quadrants(ordered[pos]):
                add(pic,f'FRAME DETAIL time={task["timestamps"][pos]:.6f}s region={name}',q.CONFIG['detail_max_pixels'])
    regions=q.motion_regions(ordered) if task['kind']=='native_contiguous' else []
    for pos in (0,len(ordered)//2,len(ordered)-1):
        pic=ordered[pos];w,h=pic.size
        for number,(a,b,c,d) in enumerate(regions):
            crop=pic.crop((int(a*w),int(b*h),int(c*w),int(d*h)))
            add(crop,f'MOTION REGION{number+1} time={task["timestamps"][pos]:.6f}s bbox={a:.3f},{b:.3f},{c:.3f},{d:.3f}',q.CONFIG['detail_max_pixels'])
    import json
    cues=q.temporal_cues(ordered)
    message.append({'type':'text','text':prompt+'\nMeasured input changes (no score targets): '+json.dumps(cues)})
    messages=[{'role':'user','content':message}]
    if previous is not None:messages.extend(q.repair_messages(previous))
    return images,messages,labels,cues,regions,reference.size


class FastNativeQwen(OriginalNativeQwen):
    def __init__(self,path):
        super().__init__(path)
        if not self.processor.image_processor.__class__.__name__.endswith('Fast'):
            raise RuntimeError('GPU preprocessing requires the pinned fast image processor')
        self.frame_cache=NativeFrameCache()

    def prepare(self,record,task,prompt,variant='original',previous=None,device='cuda'):
        started=time.perf_counter()
        ordered,content,decode=self.frame_cache.get(record,task,variant)
        mark=time.perf_counter()
        images,messages,labels,cues,regions,refsize=image_messages(record,task,ordered,prompt,previous)
        build_seconds=time.perf_counter()-mark
        mark=time.perf_counter()
        serialized=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=self.processor(text=[serialized],images=images,
            images_kwargs={'do_resize':False,'device':device},return_tensors='pt')
        if inputs['pixel_values'].device.type != device:
            raise RuntimeError('Image processor did not use the declared device')
        if device=='cuda':self.torch.cuda.synchronize()
        processor_seconds=time.perf_counter()-mark
        if len(inputs['image_grid_thw'])!=len(images):raise ValueError('Image count changed')
        token=self.processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
        count=int((inputs['input_ids']==token).sum())
        expected=int(inputs['image_grid_thw'].prod(dim=1).sum())//self.processor.image_processor.merge_size**2
        if count!=expected or count<=0:raise ValueError('Visual token count changed')
        mark=time.perf_counter()
        pixels=inputs['pixel_values'].detach().contiguous().cpu()
        effective=pixels.to(self.torch.bfloat16).view(self.torch.uint16).numpy()
        audit={'visual_input_sha256':hashlib.sha256(pixels.numpy().tobytes()).hexdigest(),
            'effective_bf16_pixels_sha256':hashlib.sha256(effective.tobytes()).hexdigest(),
            'representation':'individual timestamped images plus fixed detail crops',
            'pixel_values_shape':list(pixels.shape),'image_tokens':count,
            'image_grid_thw':inputs['image_grid_thw'].tolist(),'image_labels':labels,
            'native_content_indices':content,'temporal_cues':cues,'motion_region_boxes':regions,
            'reference_size':list(refsize),'source_frame_count':len(ordered),
            'worker_source_sha256':sha256(__file__),'preprocessing_device':device,
            'actual_pixel_device':str(inputs['pixel_values'].device),
            'timings':{**decode,'image_build_seconds':build_seconds,'processor_seconds':processor_seconds,
                'tensor_audit_seconds':time.perf_counter()-mark,'prepare_seconds':time.perf_counter()-started}}
        return inputs,audit

    def __call__(self,record,task,prompt,variant='original',previous=None):
        torch=self.torch
        started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        inputs,audit=self.prepare(record,task,prompt,variant,previous)
        inputs=inputs.to(self.model.device)
        torch.cuda.synchronize();mark=time.perf_counter()
        generation={'do_sample':False,'max_new_tokens':640 if previous is None else 1024}
        with torch.inference_mode():generated=self.model.generate(**inputs,**generation)
        torch.cuda.synchronize()
        audit.update(elapsed_seconds=time.perf_counter()-mark,
            peak_vram_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
            gpu_model=torch.cuda.get_device_name(0),generation_config=generation,seed=q.CONFIG['seed'],
            new_tokens=int(generated.shape[1]-inputs['input_ids'].shape[1]),
            quantized_language_linear_modules=self.quantized_module_count,visual_and_lm_head_quantized=False)
        raw=self.processor.batch_decode(generated[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)[0]
        audit['timings']['request_seconds']=time.perf_counter()-started
        return raw,audit


def install():
    q.VERSION=VERSION
    q.CONFIG.update(version=VERSION,max_new_tokens=640,
        fast_backend_source_sha256=sha256(__file__),frame_decoder='cached lossless RGB24 pipe, same native frame indices',
        preprocessing_device='cuda',evidence_compression='one <=25-word observation per metric; same shared anchors',
        format_repair_max_new_tokens=1024)
    q.NativeQwen=FastNativeQwen
    q.prompt_for=compact_prompt


if __name__=='__main__':
    install()
    q.main()
