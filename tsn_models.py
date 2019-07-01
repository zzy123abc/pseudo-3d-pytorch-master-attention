from torch import nn
from consensus import ConsensusModule
from attention import AttentionModule

class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model, new_length=None, consensus_type='avg', dropout=0.5):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.dropout = dropout
        self.base_model = base_model
        self.new_length = new_length
        self.consensus_type = consensus_type

        print(( """
                Initializing TSN with base model: P3D.
                TSN Configurations:
                    input_modality:     {}
                    num_segments:       {}
                    new_length:         {}
                    consensus_module:   {}
                    dropout_ratio:      {}
                """.format(self.modality, self.num_segments,
                           self.new_length, consensus_type, self.dropout) ))

        self.attention = AttentionModule()

        self.avgpool = nn.AvgPool3d(kernel_size=(self.num_segments, 5, 5), stride=1)
        self.dropout=nn.Dropout(p=dropout)
        self.fc = nn.Linear(2048, 101)

    def forward(self, input):

        base_out = self.base_model(input.view((-1,self.new_length,3) + input.size()[-2:]).permute(0,2,1,3,4))
        base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
        base_out = base_out.permute(0,2,1,3,4)

        x = self.attention(base_out)

        x = self.avgpool(x)

        x = x.view(-1,self.fc.in_features)
        x = self.fc(self.dropout(x))

        return x
