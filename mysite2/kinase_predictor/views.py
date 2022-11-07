from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django import forms
from kinase_predictor import ml_model
from kinase_predictor import GNN_explain
import rdkit
from rdkit import Chem
import pandas as pd
import os
from rdkit.Chem import PandasTools
from django.http import JsonResponse, request
import json
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
import plotly.express as px
import plotly.offline as opy
from django.http import FileResponse


# Create your views here.
from django.http import HttpResponse
import datetime


error_msg_dict = {"error_msg": None}
Base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
upload_path = os.path.join(Base_dir,r"static/file/upload_dir")
df_uniprot=pd.read_csv(os.path.join(Base_dir,r"static/file/UniProt.csv"))
df_uniprot.columns=["UniProt","Group","Family","Name","Symbol"]



@csrf_exempt
def download(request):
    #https://www.zhangshengrong.com/p/Z9a28xMkXV/
    # if request.is_ajax():
    if True:
        if request.method == "POST":
            # mode = data["mode"]
            # index = data["index"]
            mode = request.POST.get("mode")
            index = request.POST.get("index", "")
            file_index = request.POST.get("file_index", "")#single_mol
            print("mode",mode,"index",index,"file_index",file_index)


            if mode=="gobal_result":
                grobal_path = os.path.join(Base_dir,r"static/file/download_file/grobal_result/result_%d.csv"%(int(index)))
                file=open(grobal_path,'rb')
                response =FileResponse(file)
                response['Content-Type']='text/csv'
                response['Content-Disposition']='attachment;filename="result.csv"'
                print("response----->",response)
                return response

            if mode=="invalid_smiles":
                invalid_path=os.path.join(Base_dir,r"static/file/download_file/invalid_smiles/invalid_smiles_%d.csv"%(int(index)))
                file=open(invalid_path,'rb')
                response =FileResponse(file)
                response['Content-Type']='text/csv'
                response['Content-Disposition']='attachment;filename="invalid_smiles.csv"'
                print("response----->",response)
                return response
            
            if mode=="single_mol":
                invalid_path=os.path.join(Base_dir,r"static/file/download_file/single_mol/result_%d.csv"%(int(index)))
                file=open(invalid_path,'rb')
                response =FileResponse(file)
                response['Content-Type']='text/csv'
                response['Content-Disposition']='attachment;filename=%s'%("result_%d.csv"%(int(file_index)))
                print("response----->",response)
                return response



def homepage(request):
    return render(request, 'homepage.html')


def submit(request):
    return render(request, 'submit.html')

# @csrf_exempt
def result(request):
    dic = {'status': 201, 'error_msg': None}

    print("ajax",request.is_ajax())
    if request.is_ajax():

        if request.method == "POST":
            smiles = None
            file = request.FILES.get("file")

            print(file,"file")

            # 判断文件是否能存在
            if file:
                file_name = file.name
                file_type = file_name.split(".")[-1]
                print("file", file)
                print("file_name", file_name)
                print("file_type", file_type)
                file_path = os.path.join(Base_dir, r"static/file/upload_dir", file_name)

                # 先把文件写入
                with open(file_path, "wb") as f:
                    for line in file:
                        f.write(line)

                #检查sdf文件
                if file_type == "sdf":
                    suppl = Chem.SDMolSupplier(file_path)
                    mols = [Chem.MolToSmiles(mol) for mol in suppl if mol]

                    #如果sdf文件中没有合法的分子就舍弃
                    if len(mols)>0:
                        dic["status"] = 200
                    # 如果sdf文件中没有一个合法的分子
                    elif len(mols)>100000:
                        dic["error_msg"] = "Too much molecule (>100000) in the file for computation"
                    else:
                        dic["error_msg"] = "SDF file must contain at least one valid mol"


                #检查csv文件
                elif file_type == "csv":

                    df0 = pd.read_csv(file_path)
                    # 如果文件中不包含smiles列，就返回错误
                    if "smiles" not in df0.columns:
                        dic["error_msg"] = "A column with the header named \' smiles \' not in CSV file!"

                    else:
                        PandasTools.AddMoleculeColumnToFrame(df0, smilesCol='smiles')
                        df0 = df0[~df0.ROMol.isnull()]
                        PandasTools.RemoveSaltsFromFrame(df0)
                        del df0["ROMol"]

                        #如果文件中没有一个合法的分子
                        if len(df0)==0:
                            dic["error_msg"] = "CSV file must contain at least one valid molecule"
                        elif len(df0)>100000:
                            dic["error_msg"] = "Too much molecule (>100000) in the file for computation"
                        else:
                            dic["status"] = 200

                        df0 = None

                #双重保险
                else:
                    dic["error_msg"]="Input file is not sdf file or csv file"

            #smiles 模式
            else:
                data = json.loads(request.body)
                if "drawing" in data:
                    drawing = data["drawing"]
                    print("drawing",drawing)
                    drawing_path = os.path.join(Base_dir, r"static/file/upload_dir/drawing.mol")

                    # 先把文件写入
                    with open(drawing_path, "w") as f:
                        for line in drawing:
                            f.write(line)
                    
                    try:
                        mol = Chem.MolFromMolFile(drawing_path)
                        print("drawing",Chem.MolToSmiles(mol))
                        dic["status"] = 200

                    except:
                        dic["error_msg"] = "please draw a correct molecule"
                    
                else:
                    smiles = data["smiles"]
                    print("smiles-->",smiles)
                    mol = Chem.MolFromSmiles(smiles)
                    try:
                        smiles1 = Chem.MolToSmiles(mol)
                        dic["status"] = 200

                    except:
                        dic["error_msg"] = "please input correct smiles"

            return HttpResponse(json.dumps(dic))


    else:
        result_dict = {"invalid_smiles": None,"index_result":None, "result": [],"smiles_list":[],"mol_graph":[]}
        if request.method == 'POST':

            file = request.FILES.get("file")
            smiles = request.POST.get("smiles", "")
            drawing = request.POST.get("drawing", "")
            print("smiles",smiles,"file",file)
            print("drawing",drawing)
            if drawing:

                drawing_path = os.path.join(Base_dir, r"static/file/upload_dir/drawing1.mol")
                

                # 先把文件写入
                with open(drawing_path, "w") as f:
                    for line in drawing:
                        f.write(line)

                mol = Chem.MolFromMolFile(drawing_path)
                smiles = Chem.MolToSmiles(mol)
                print("smiles",smiles)


            try:
                file_name = file.name
                file_type = file_name.split(".")[-1]
                print("file_name", file_name, "file_type", file_type)
                file_path = os.path.join(Base_dir, r"static/file/upload_dir", file_name)
                # 先把文件写入
                with open(file_path, "wb") as f:
                    for line in file:
                        f.write(line)


                if file_type == "sdf":
                    suppl = Chem.SDMolSupplier(file_path)
                    mols = [Chem.MolToSmiles(mol) for mol in suppl if mol]
                    df = pd.DataFrame(mols, columns=["smiles"])

                elif file_type == "csv":
                    df0 = pd.read_csv(file_path)
                    df = pd.DataFrame(df0["smiles"])
            except:
                df = pd.DataFrame([smiles], columns=["smiles"])


            result_df,index_result,invalid_smiles,length_invilid_fold = ml_model.model_output(df)
            result_dict["smiles_list"]=[_ for _ in result_df["canonical_smiles"]]
            print("check",result_dict["smiles_list"])
            for _,smi in enumerate(result_dict["smiles_list"]):
                mol = Chem.MolFromSmiles(smi)
                d2d = Draw.MolDraw2DSVG(300, 210)
                d2d.DrawMolecule(mol)
                d2d.FinishDrawing()
                text = d2d.GetDrawingText()
                text = text.replace("fill:#FFFFFF;","fill:transparent;")
                text_index=text.find(r"<!-- END OF HEADER -->")-2
                text = text[:text_index]+"style=' width: 100%; max-width: 300px; height: auto; '"+text[text_index:]
                result_dict["mol_graph"].append(text)


            result_dict["result"]=result_df.values[:,-204:].tolist()

            return_dict={"item_result": [{'smiles': t[0],"mol_graph": t[1],'result': t[2]} for t in zip(result_dict["smiles_list"], result_dict["mol_graph"],result_dict["result"])],}
            return_dict["index_result"] = index_result
            return_dict["invalid_smiles"] = invalid_smiles
            return_dict["length_invilid_fold"] = length_invilid_fold
            print("return_dict",return_dict)
            return render(request,'result.html', return_dict)


def molecule(request):
    print("body-->",request.body)
    print("request.is_ajax()",request.is_ajax())
    #pass

    #explain part
    if request.is_ajax():
        if request.method == "POST":
            data = json.loads(request.body)
            print("data",data)
            smiles = data["smiles"]
            mode="group"

            if "group" in data:
                group = data["group"]
                print("group",group)
                mode_select=float(data["mode_select"])
                df_explain = df_uniprot.copy()

                predict_result_str=data["predict_result_str"]
                print("predict_result_str",predict_result_str)
                predict_result=[round(float(_),3) for _ in predict_result_str.split(",")]
                print("predict_result",predict_result)
                df_explain["predict_result"]=predict_result
                df_explain=df_explain[df_explain["predict_result"]>=mode_select]
                del df_explain["predict_result"]

                if group!="All":
                    df_explain = df_explain[df_explain["Group"]==group]
                

            else:
                uniprot = data["uniprot"]
                print("uniprot",uniprot)
                df_explain = df_uniprot.copy()
                df_explain = df_explain[df_explain["UniProt"]==uniprot]
                mode="uniprot"

            iter_result=[]
            for _ in df_explain.index:
                uniprot = df_explain.loc[_,"UniProt"]
                group = df_explain.loc[_,"Group"]
                family = df_explain.loc[_,"Family"]
                name = df_explain.loc[_,"Name"]
                symbol = df_explain.loc[_,"Symbol"]
                explain,prediction=GNN_explain.explain_GNN(smiles,uniprot,mode)
                iter_result.append({
                    "uniprot":uniprot,
                    "group":group,
                    "name":name,
                    "symbol":symbol,
                    "family":family,
                    "explain":explain,
                    "prediction":round(prediction,3),#如果有需要请在此处添加
                })

            return HttpResponse(json.dumps(iter_result))


    else:
        if request.method == 'POST':
            smiles = request.POST.get("smiles", "")
            result = request.POST.get("result", "")
            index = request.POST.get("index", "")

            
            result_float=[float(_) for _ in result[1:-1].split(",")]

            df_result = df_uniprot.copy()
            df_result["Prediction"] = np.array(result_float)


            # group_div
            df_result["Count"] = np.array(np.array(df_result["Prediction"]) >= 0.5).astype(np.int)
            df_result.columns=["UniProt","Group","Family","Name","Symbol","Prediction","Count"]

            df_result_count = pd.DataFrame(df_result.groupby("Group").sum()["Count"])
            df_result_count = df_result_count.reset_index()

            group_array=[_ for _ in df_result_count["Group"]]
            group_count=[_ for _ in df_result_count["Count"]]

            

            # uniprot_div
            uniprot_fig = px.sunburst(
                df_result,
                  path=["Group","Family","UniProt"],
                  values="Prediction",
                #   width=800, height=800,
                  color_continuous_scale="reds",
                  color="Prediction",
                  branchvalues="total",
            )
            # uniprot_fig.update_layout(autosize=True)
            # uniprot_fig.update_layout(
            #     title={
            #         'text': "Overall Inhibitory Activity against Kinases",
            #         'y':0.95,
            #         'x':0.5,
            #         'xanchor': 'center',
            #         'yanchor': 'top'},
            #     legend_title="Legend Title",
            # )

            # uniprot_div = opy.plot(uniprot_fig, auto_open=False, output_type='div')
            # print("uniprot_div",uniprot_div)
            uniprot_div = uniprot_fig.to_html(full_html=False, default_height=711, default_width=728)
      


            mol = Chem.MolFromSmiles(smiles)
            d2d = Draw.MolDraw2DSVG(360, 324)
            d2d.DrawMolecule(mol)
            d2d.FinishDrawing()
            mol_graph = d2d.GetDrawingText()
            mol_graph = mol_graph.replace("fill:#FFFFFF;","fill:transparent;")
            #mol_graph_index=mol_graph.find(r"<!-- END OF HEADER -->")-2
            #mol_graph = mol_graph[:mol_graph_index]+"style=' width: 100%; max-width: 480px; height: auto; '"+mol_graph[mol_graph_index:]

            result_list=[]
            for _ in df_result.index:
                uniprot=df_result.loc[_,"UniProt"]
                group=df_result.loc[_,"Group"]
                family=df_result.loc[_,"Family"]
                prediction=round(df_result.loc[_,"Prediction"],3)
                name = df_result.loc[_,"Name"]
                symbol = df_result.loc[_,"Symbol"]
                result_list.append(
                    {'uniprot': uniprot,"group": group,'family': family,"name":name,"symbol":symbol,"prediction":prediction,}
                )
            
            predict_result=[round(_,3) for _ in df_result["Prediction"]]
            predict_result_str=""
            for idx,num in enumerate(predict_result):
                if idx<len(predict_result)-1:
                    predict_result_str+=str(num)+","
                else:
                    predict_result_str+=str(num)

             
            df_download = df_result[["UniProt","Group","Family","Name","Symbol","Prediction"]]
            df_download["Canonical_Smiles"]=smiles
            df_download = df_download[["Canonical_Smiles","UniProt","Group","Family","Name","Symbol","Prediction"]]
            df_download["Prediction"]=df_download["Prediction"].round(decimals=4)
            single_length = len(os.listdir(os.path.join(Base_dir,r"static/file/download_file/single_mol")))
            df_download.to_csv(os.path.join(Base_dir,r"static/file/download_file/single_mol/result_%d.csv"%(single_length)),index=False)

            return render(request, 'molecule.html',{"index":index,
                                                    "smiles":smiles,
                                                    "mol_graph":mol_graph,
                                                    "group_array":group_array,
                                                    "group_count":group_count,
                                                    "uniprot_div":uniprot_div,
                                                    "result":result_list,
                                                    "predict_result_str":predict_result_str,
                                                    "single_length":single_length,
                                                    })



def help(request):

    return render(request, 'help.html')


def contact(request):

    return render(request, 'contact.html')

def trysth(request):

    return render(request, 'trysth.html')

