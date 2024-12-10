import torch

x=torch.randn(1, 784)
w=torch.randn(10, 784)

logits=x@w.t()

pred=torch.softmax(logits, dim=1)
pred_log=torch.log(pred)
print(pred_log)

ce_ttma=torch.tensor([-pred_log[0, 3]])

target=torch.tensor([3])

ce_manu=torch.nn.functional.nll_loss(pred_log, target)

ce_auto=torch.nn.functional.cross_entropy(logits, target)

print('ce_manu==ce_auto:', torch.equal(ce_manu, ce_auto))

print(ce_ttma, ce_manu, ce_auto)







x=torch.randn(3, 784)
w=torch.randn(2, 784)

logits=x@w.t()

pred=torch.softmax(logits, dim=1)
pred_log=torch.log(pred)
print(pred_log)



target=torch.tensor([0, 1, 0])

# ce_ttma=-pred_log.index_select(1, target)
ce_ttma=torch.tensor([-pred_log[0, 0], -pred_log[1, 1], -pred_log[2, 0]]).sum()/len(target)

ce_manu=torch.nn.functional.nll_loss(pred_log, target)

ce_auto=torch.nn.functional.cross_entropy(logits, target)

print('ce_manu==ce_auto:', torch.equal(ce_manu, ce_auto))

print(ce_ttma, ce_manu, ce_auto)

