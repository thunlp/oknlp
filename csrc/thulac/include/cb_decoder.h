#ifndef __DP_H__
#define __DP_H__
#include<cstdlib>

namespace permm{


struct Node{
    int type;
    int* predecessors;
    int* successors;
};

struct Alpha_Beta{
    int value;
    int node_id;
    int label_id;
};

/** The DP algorithm(s) for path labeling */
inline int dp_decode(
        int l_size,
        int* ll_weights,
        int node_count,
        Node* nodes,
        int* values,
        Alpha_Beta* alphas,
        int* result,
        int** pre_labels=NULL,
        int** allowed_label_lists=NULL
        ){
    //calculate alphas
    int node_id;
    int* p_node_id;
    int* p_pre_label;
    int* p_allowed_label;
    int k;
    int j;
    Alpha_Beta* tmp;
    Alpha_Beta best;best.node_id=-1;
    Alpha_Beta* pre_alpha;
    int score;
    
    for(int i=0;i<node_count*l_size;i++)alphas[i].node_id=-2;
    for(int i=0;i<node_count;i++){//for each node
        p_allowed_label=allowed_label_lists?allowed_label_lists[i]:NULL;
        j=-1;
        int max_value=0;
        int has_max_value=0;

        while((p_allowed_label?
                    ((j=(*(p_allowed_label++)))!=-1):
                    ((++j)!=l_size))){
            if((!has_max_value) || (max_value<values[i*l_size+j])){
                has_max_value=1;
                max_value=values[i*l_size+j];
            }
        }
        p_allowed_label=allowed_label_lists?allowed_label_lists[i]:NULL;
        j=-1;
        while((p_allowed_label?
                    ((j=(*(p_allowed_label++)))!=-1):
                    ((++j)!=l_size))){
            
            tmp=&alphas[i*l_size+j];
            tmp->value=0;
            p_node_id=nodes[i].predecessors;
            p_pre_label=pre_labels?pre_labels[j]:NULL;
            while((node_id=*(p_node_id++))>=0){
                k=-1;
                while(p_pre_label?
                        ((k=(*p_pre_label++))!=-1):
                        ((++k)!=l_size)
                        ){
                    pre_alpha=alphas+node_id*l_size+k;
                    if(pre_alpha->node_id==-2)continue;
                    score=pre_alpha->value+ll_weights[k*l_size+j];
                    if((tmp->node_id<0)||(score>tmp->value)){
                        tmp->value=score;
                        tmp->node_id=node_id;
                        tmp->label_id=k;
                    }
                }
            }
            tmp->value+=values[i*l_size+j];
            
            if((nodes[i].type==1)||(nodes[i].type==3))
                tmp->node_id=-1;
            if(nodes[i].type>=2){
                if((best.node_id==-1)||(best.value<tmp->value)){
                    best.value=tmp->value;
                    best.node_id=i;
                    best.label_id=j;
                }
            }
        }
    }

    tmp=&best;
    while(tmp->node_id>=0){
        result[tmp->node_id]=tmp->label_id;
        tmp=&(alphas[(tmp->node_id)*l_size+(tmp->label_id)]);
    }
    return best.value;
};


}
#endif
