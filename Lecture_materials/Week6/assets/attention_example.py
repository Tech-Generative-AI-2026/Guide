import torch
from torch import nn
torch.set_printoptions(precision=4,sci_mode=False)
x=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
wq=torch.tensor([[1.,0.],[0.,1.]])
wk=torch.tensor([[1.,1.],[0.,1.]])
wv=torch.tensor([[1.,2.],[3.,0.]])
q,k,v=x@wq,x@wk,x@wv
s=q@k.T / 2**.5
a=s.softmax(-1)
mask=torch.triu(torch.ones(3,3,dtype=torch.bool),diagonal=1)
c=s.masked_fill(mask,-torch.inf).softmax(-1)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in=2, d_out=4, num_heads=2, context_length=3, dropout=0.5):
        super().__init__()
        if d_out % num_heads: raise ValueError('d_out must be divisible by num_heads')
        self.num_heads=num_heads
        self.head_dim=d_out//num_heads
        self.d_out=d_out
        self.W_query=nn.Linear(d_in,d_out,bias=False)
        self.W_key=nn.Linear(d_in,d_out,bias=False)
        self.W_value=nn.Linear(d_in,d_out,bias=False)
        self.out_proj=nn.Linear(d_out,d_out,bias=False)
        self.dropout=nn.Dropout(dropout)
        self.register_buffer('mask',torch.triu(torch.ones(context_length,context_length,dtype=torch.bool),diagonal=1))
    def forward(self,x):
        b,t,_=x.shape
        if t>self.mask.shape[0]: raise ValueError('context_length exceeded')
        q=self.W_query(x).view(b,t,self.num_heads,self.head_dim).transpose(1,2)
        k=self.W_key(x).view(b,t,self.num_heads,self.head_dim).transpose(1,2)
        v=self.W_value(x).view(b,t,self.num_heads,self.head_dim).transpose(1,2)
        scores=q@k.transpose(-2,-1)/self.head_dim**.5
        weights=scores.masked_fill(self.mask[:t,:t],-torch.inf).softmax(-1)
        context=self.dropout(weights)@v
        joined=context.transpose(1,2).contiguous().view(b,t,self.d_out)
        return self.out_proj(joined)

# 슬라이드의 두 head와 항등 출력 변환을 재현하는 공통 실행 상태.
batch = torch.stack((x, x), dim=0)
m = MultiHeadAttention()
with torch.no_grad():
    m.W_query.weight.copy_(torch.cat((wq, wq.flip(1)), dim=1).T)
    m.W_key.weight.copy_(torch.cat((wk, wk), dim=1).T)
    m.W_value.weight.copy_(torch.cat((wv, wv), dim=1).T)
    m.out_proj.weight.copy_(torch.eye(4))
m.eval()
y = m(batch)
changed = batch.clone()
changed[:, 2] = torch.tensor([9., -4.])
y_changed = m(changed)
assert torch.allclose(y[:, :2], y_changed[:, :2])
assert not torch.allclose(y[:, 2], y_changed[:, 2])
if __name__ == '__main__':
    print('output shape:', tuple(y.shape))
    print('second token:', y[0, 1].detach())
    print('causal invariance: PASS')
