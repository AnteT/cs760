from dataclasses import dataclass
from math import log2

@dataclass
class DataRow:
    X:bool
    Y:bool
    Z:bool
    count:int

datarow_dict = {
 1: DataRow(True, True, True, 36)
,2: DataRow(True, True, False, 4)
,3: DataRow(True, False, True, 2)
,4: DataRow(True, False, False, 8)
,5: DataRow(False, True, True, 9)
,6: DataRow(False, True, False, 1)
,7: DataRow(False, False, True, 8)
,8: DataRow(False, False, False, 32)
}

raw_data = (
 (True, True, True, 36)
,(True, True, False, 4)
,(True, False, True, 2)
,(True, False, False, 8)
,(False, True, True, 9)
,(False, True, False, 1)
,(False, False, True, 8)
,(False, False, False, 32)
)

X_True, X_False, Y_True, Y_False, Z_True, Z_False = 0, 0, 0, 0, 0, 0

for row in raw_data:
    X, Y, Z, count = row[0], row[1], row[2], row[3]
    if X:
        X_True += count
    elif X == False:
        X_False += count
    if Y:
        Y_True += count
    elif Y == False:
        Y_False += count        
    if Z:
        Z_True += count
    elif Z == False:
        Z_False += count

def mutual_info(conditions:str) -> dict:
    """use X,Y,Z for targets c1 and c2"""
    conditions = conditions.upper()
    if 'X' in conditions and 'Y' in conditions:
        conditions = 'XY'
    elif 'X' in conditions and 'Z' in conditions:
        conditions = 'XZ'
    elif 'Y' in conditions and 'Z' in conditions:
        conditions = 'YZ'
    else:
        print(f'unrecognized conditions: {conditions}')
        return {}
    print(f'finding mutal information for {conditions}')    
    sum_c1_T, sum_c2_T = 0,0
    sum_c1_F, sum_c2_F = 0,0
    sum_TT, sum_TF, sum_FT, sum_FF = 0,0,0,0
    for v in datarow_dict.values():
        match conditions:
            case 'XY':
                if v.X:
                    sum_c1_T += v.count
                else:
                    sum_c1_F += v.count
                if v.Y:
                    sum_c2_T += v.count
                else:
                    sum_c2_F += v.count
                if v.X and v.Y:
                    sum_TT += v.count
                elif v.X and not v.Y:
                    sum_TF += v.count
                elif not v.X and v.Y:
                    sum_FT += v.count
                elif not v.X and not v.Y:
                    sum_FF += v.count
            case 'XZ':
                if v.X:
                    sum_c1_T += v.count
                else:
                    sum_c1_F += v.count
                if v.Z:
                    sum_c2_T += v.count                
                else:
                    sum_c2_F += v.count                
                if v.X and v.Z:
                    sum_TT += v.count
                elif v.X and not v.Z:
                    sum_TF += v.count
                elif not v.X and v.Z:
                    sum_FT += v.count
                elif not v.X and not v.Z:
                    sum_FF += v.count 
            case 'YZ':
                if v.Y:
                    sum_c1_T += v.count
                else:
                    sum_c1_F += v.count
                if v.Z:
                    sum_c2_T += v.count                  
                else:
                    sum_c2_F += v.count                  
                if v.Y and v.Z:
                    sum_TT += v.count
                elif v.Y and not v.Z:
                    sum_TF += v.count
                elif not v.Y and v.Z:
                    sum_FT += v.count
                elif not v.Y and not v.Z:
                    sum_FF += v.count
    c1, c2 = conditions[0], conditions[1]
    P1T = sum_c1_T/100
    P1F = sum_c1_F/100
    P2T = sum_c2_T/100
    P2F = sum_c2_F/100
    PTT = sum_TT/100
    PTF = sum_TF/100
    PFT = sum_FT/100
    PFF = sum_FF/100
    probs_dict = {
         f'params': conditions
        ,f'P({c1}=T)': P1T
        ,f'P({c1}=F)': P1F
        ,f'P({c2}=T)': P2T
        ,f'P({c2}=F)': P2F
        ,f'P({c1}=T, {c2}=T)': PTT
        ,f'P({c1}=T, {c2}=F)': PTF
        ,f'P({c1}=F, {c2}=T)': PFT
        ,f'P({c1}=F, {c2}=F)': PFF
    }
    mi = (PTT * (log2(PTT/(P1T*P2T)))) + (PTF * (log2(PTF/(P1T*P2F)))) + (PFT * (log2(PFT/(P1F*P2T)))) + (PFF * (log2(PFF/(P1F*P2F))))
    probs_dict.update({f'I({c1}, {c2})': round(mi,6)})
    return probs_dict

for q in ('XY', 'XZ', 'ZY'):
    probs_dict = mutual_info(q)
    params = probs_dict['params']
    c1, c2 = params[0], params[1]
    mi_key = f'I({c1}, {c2})'
    print(f"{mi_key} = {probs_dict[mi_key]}")
