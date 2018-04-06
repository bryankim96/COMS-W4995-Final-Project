require 'cunn'
require 'cutorch'
require 'cudnn'
require 'image'
require 'nn'
require 'nngraph'
require 'lfs'
torch.setdefaulttensortype('torch.FloatTensor')

local doc_length = 201

local alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
local dict = {}
for i = 1,#alphabet do
    dict[alphabet:sub(i,i)] = i
end
ivocab = {}
for k,v in pairs(dict) do
  ivocab[v] = k
end

opt = {
  query_str='',
  embedding_net='',
  gpu=0,
}

print(opt.query_str)
print(opt.embedding_net)

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

net_txt = torch.load(opt.embedding_net)
if net_txt.protos ~=nil then net_txt = net_txt.protos.enc_doc end

net_txt:evaluate()

local txt = torch.zeros(1,doc_length,#alphabet)
for t = 1,doc_length do
  local ch = opt.query_str:sub(t,t)
  local ix = dict[ch]
  if ix ~= 0 and ix ~= nil then
    txt[{1,t,ix}] = 1
    txt = txt:cuda()
  end
end

if opt.gpu > 0 then
  require 'cunn'
  require 'cudnn'
  net_gen:cuda()
  net_txt:cuda()
  noise = noise:cuda()
end

raw_txt = opt.query_str
fea_txt = net_txt:forward(txt):clone():float()

torch.save('test_embedding.t7', fea_txt)

