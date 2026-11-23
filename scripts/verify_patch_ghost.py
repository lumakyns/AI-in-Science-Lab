"""Run from repository root: python -m scripts.verify_patch_ghost."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.surround_attention.models.attention import LocalAttention


def main():
    print('torch',torch.__version__,'CUDA',torch.version.cuda,'available',torch.cuda.is_available(),flush=True)
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    for experiment in ('predictive_ghost','lifetime_ghost'):
        trainer = importlib.import_module(f'src.{experiment}.training')
        attention = importlib.import_module(f'src.{experiment}.models.attention').SelectedAttention
        routing = importlib.import_module(f'src.{experiment}.models.routing').PatchBlock
        for family in ('nat','vil'):
            torch.manual_seed(0)
            ref = LocalAttention(8,2,4,3,family,1).double()
            actual = attention(8,2,4,3,family,int(family=='vil')).double()
            actual.load_state_dict(ref.state_dict())
            x = torch.randn(2,16+actual.globals,8,dtype=torch.double,requires_grad=True)
            mask = torch.rand(2,16) > .5
            global_out,spatial = actual(x,mask)
            full = ref(x)
            torch.testing.assert_close(spatial[mask],full[:,actual.globals:][mask],rtol=1e-8,atol=1e-10)
            torch.testing.assert_close(global_out,full[:,:actual.globals],rtol=1e-8,atol=1e-10)
            a=spatial.square().sum()+global_out.square().sum()
            b=full[:,actual.globals:][mask].square().sum()+full[:,:actual.globals].square().sum()
            torch.testing.assert_close(torch.autograd.grad(a,x,retain_graph=True)[0],torch.autograd.grad(b,x)[0],rtol=1e-8,atol=1e-10)
            # All-empty masks can occur for individual lifetime evaluation images.
            empty_global,empty_spatial = actual(x,torch.zeros_like(mask))
            assert empty_spatial.count_nonzero()==0 and torch.isfinite(empty_global).all()
            c=trainer.load_config(f'src/{experiment}/config.yaml')
            c['model']['family']=family
            for policy in ('dense','random','predictive','lifetime'):
                c['model']['routing']['policy']=policy
                torch.manual_seed(0)
                model=trainer.GhostTransformer(dataset=trainer.Dataset.CIFAR100,**c['model']).cuda()
                batch = 128 if policy == 'dense' else 8
                scaler=torch.amp.GradScaler('cuda')
                opt=torch.optim.AdamW(model.parameters(),lr=.001)
                with torch.autocast('cuda',dtype=torch.float16):
                    logits=model(torch.randn(batch,3,32,32,device='cuda'))
                    loss=torch.nn.functional.cross_entropy(logits,torch.arange(batch,device='cuda')%100)+.1*model.auxiliary()
                assert torch.isfinite(loss)
                scaler.scale(loss).backward();scaler.step(opt);scaler.update()
                assert model.routing_blocks()[0].predictor.weight.grad is not None
                if policy != 'dense':
                    assert model.routing_blocks()[0].ghost.weight.grad.abs().sum() > 0
                for layer in model.routing_blocks():
                    if policy == 'lifetime':
                        assert layer.last_mask is not None and layer.calibrated
                model.eval()
                with torch.no_grad():
                    x2=torch.randn(2,3,32,32,device='cuda')
                    y2=model(x2)
                    if policy == 'lifetime':
                        y1=model(x2[:1]); torch.testing.assert_close(y1,y2[:1],atol=2e-5,rtol=2e-4)
                print(experiment,family,policy,'CUDA forward/backward and evaluation passed',flush=True)
                del model,opt,logits,loss
                torch.cuda.empty_cache()
            # Full synthetic training/output/explicit offline-tracking smoke test.
            c['model']['routing']['policy']='lifetime' if experiment=='lifetime_ghost' else 'predictive'
            c['training'].update(epochs=1,log_every_batches=1)
            c['scheduler']['warmup_epochs']=0
            c['wandb']['mode']='offline'
            synthetic={split:DataLoader(TensorDataset(torch.randn(8,3,32,32),torch.arange(8)),batch_size=4) for split in ('train','validation','test')}
            directory=Path(c['checkpoint']['dir'])/'smoke'/trainer.TIMESTAMP/family
            trainer.run_experiment(c,directory,data=synthetic)
            state=torch.load(directory/'checkpoints/last.pt',weights_only=True,map_location='cpu')
            restored=trainer.GhostTransformer(dataset=trainer.Dataset.CIFAR100,**c['model'])
            restored.load_state_dict(state['model'],strict=True)
            assert (directory/'COMPLETE').is_file() and (directory/'images/epoch_001.png').is_file()
            result=json.loads((directory/'result.json').read_text())
            assert result['test/examples']==8 and 0<=result['test/query_fraction']<=1
        block=routing(8,2,4,3,'nat',0,2,0,dict(policy='lifetime',keep_ratio=.25,threshold_decay=.95))
        errors=torch.arange(128).reshape(8,16).float()
        mask=block.select(errors)
        assert (mask.sum(0)==2).all() and mask[-2:].all() and not mask[:-2].any()
        block.eval()
        torch.testing.assert_close(block.select(errors[:1]),block.select(errors)[:1])
    print('All attention references, gradients, routing semantics, actual-device, and pipeline checks passed.',flush=True)


if __name__=='__main__':
    main()
