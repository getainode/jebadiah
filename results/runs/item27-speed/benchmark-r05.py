import sys, os, json, time, random, gc, traceback
from pathlib import Path
sys.path.insert(0,'/workspace/src/train')
import torch
from transformers import TrainingArguments,TrainerCallback
from huggingface_hub import HfApi,snapshot_download
from peft import get_peft_model,LoraConfig
from jebadiah_model import load_base,load_tokenizer,option_logits
from jebadiah_prompt import Renderer
from train_jebadiah import DecideDataset,DecideCollator,DecideTrainer,FiniteGuard
from hub_checkpoints import HubCheckpoint
from cuda_preflight import main as preflight

def emit(x): print('ITEM27 '+json.dumps(x),flush=True)
preflight()
base=snapshot_download('Qwen/Qwen3.5-9B',revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a',allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','*.model'])
data=snapshot_download('frontier-infra/jebadiah-data-v2-1',repo_type='dataset',revision='a8e69f2dcab2ead8b164259d0c69102c5cc50e41',allow_patterns=['train.jsonl'])
ds=DecideDataset([json.loads(l) for l in open(Path(data)/'train.jsonl') if l.strip()]); tok=load_tokenizer(base); renderer=Renderer(tok,4096)
model=get_peft_model(load_base(base),LoraConfig(r=64,lora_alpha=128,lora_dropout=.05,bias='none',target_modules='all-linear',task_type='CAUSAL_LM'))
initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
api=HfApi(); report=[]
class Clock(TrainerCallback):
 def __init__(self):self.times=[]
 def on_step_begin(self,args,state,control,**kwargs):
  torch.cuda.synchronize();self.start=time.perf_counter()
  if state.global_step==21:
   self.prof=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]);self.prof.start()
 def on_step_end(self,args,state,control,**kwargs):
  torch.cuda.synchronize();self.times.append(time.perf_counter()-self.start)
  if state.global_step==22:
   self.prof.stop();self.prof.export_chrome_trace('/workspace/item27-'+tag+'.trace.json')
   events=self.prof.key_averages()
   top=sorted(events,key=lambda e:e.self_device_time_total,reverse=True)[:20]
   emit({'tag':tag,'profile_top':[{'name':e.key,'calls':e.count,'self_cpu_us':e.self_cpu_time_total,'self_cuda_us':e.self_device_time_total} for e in top]})
  if state.global_step%10==0: emit({'tag':tag,'step':state.global_step,'last_step_s':self.times[-1]})
class CountCollator(DecideCollator):
 def __init__(self,*a,**kw):super().__init__(*a,**kw);self.real=0;self.padded=0;self.cpu_s=0
 def __call__(self,rows):
  t=time.perf_counter();batch=super().__call__(rows);self.cpu_s+=time.perf_counter()-t
  self.real+=int(batch['attention_mask'].sum());self.padded+=batch['input_ids'].numel();return batch
for tag,micro,accum,checkpoint,grouped in [('baseline',4,2,True,False),('autocast-adaptive',8,1,True,True)]:
 repo='frontier-infra/jebadiah-9b-item27-r05-'+tag+'-checkpoints';api.create_repo(repo,private=True,exist_ok=False)
 cfg={'base_identity':'Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a','resume':'none','checkpoint_repo':repo,'batch_size':micro,'gradient_accumulation_steps':accum,'use_gradient_checkpointing':checkpoint,'group_by_length':grouped,'pad_to_multiple_of':64 if grouped else None,'checkpoint_min_tokens':1024 if grouped else None,'backbone_autocast':grouped,'lora_rank':64}
 try:
  model.zero_grad(set_to_none=True)
  for n,p in model.named_parameters():
   if p.requires_grad:p.data.copy_(initial[n].to(p.device))
  if checkpoint:model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
  else:model.gradient_checkpointing_disable();model.disable_input_require_grads()
  if grouped:ds.measure_lengths(renderer)
  args=TrainingArguments(output_dir='/workspace/item27-'+tag,max_steps=150,per_device_train_batch_size=micro,gradient_accumulation_steps=accum,learning_rate=1e-4,weight_decay=0.,warmup_steps=0,lr_scheduler_type='cosine',bf16=True,logging_steps=10,logging_nan_inf_filter=False,save_strategy='steps',save_steps=10000,save_total_limit=2,report_to=[],seed=17,dataloader_num_workers=0,remove_unused_columns=False,gradient_checkpointing=False,optim='adamw_torch',max_grad_norm=1.,train_sampling_strategy='group_by_length' if grouped else 'random')
  clock=Clock();collator=CountCollator(tok,renderer,True,score_targets='ordinal',pad_to_multiple_of=64 if grouped else None)
  trainer=DecideTrainer(model=model,args=args,train_dataset=ds,data_collator=collator,callbacks=[FiniteGuard(),clock,HubCheckpoint(cfg)],checkpoint_min_tokens=cfg.get('checkpoint_min_tokens'),backbone_autocast=cfg.get('backbone_autocast',False))
  trainer.model_accepts_loss_kwargs=False
  torch.cuda.reset_peak_memory_stats();t=time.perf_counter();trainer.train(resume_from_checkpoint=None);wall=time.perf_counter()-t
  times=clock.times[50:]
  result={'tag':tag,'recipe':cfg,'steps':len(clock.times),'warmup_steps':50,'questions_per_second':len(times)*8/sum(times),'step_times':clock.times,'peak_gb':torch.cuda.max_memory_allocated()/1e9,'train_wall_s':wall,'real_tokens':collator.real,'padded_tokens':collator.padded,'padding_fraction':1-collator.real/collator.padded,'cpu_collation_s':collator.cpu_s}
  emit(result);report.append(result)
  api.upload_file(path_or_fileobj='/workspace/item27-'+tag+'.trace.json',path_in_repo='item27-step.trace.json',repo_id=repo)
  api.upload_file(path_or_fileobj=json.dumps(result,indent=2).encode(),path_in_repo='item27-production-benchmark.json',repo_id=repo)
  trainer=None;gc.collect();model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
 except Exception as e:emit({'tag':tag,'error':str(e),'traceback':traceback.format_exc()});model.zero_grad(set_to_none=True);trainer=None;gc.collect();torch.cuda.empty_cache()
emit({'done':True,'results':report})
