## SEGMENTAÇÃO DE IMAGENS ##

Esse trabalho consiste na segmentação semântica de imagens de OCT (Optical Coherence Tomography)
com o intuito de facilitar a detecção das camadas que compõem a retina e as doenças que acometem
essa área ocular.

# 1 - dataset usado
o dataset consiste num conjunto de dados de 10 pacientes arquivados em documentos .mat. Esses ar-
quivos basicamente são dicionários python, em que contém informações dos pacientes, como suas imagens
e as anotações que os médicos especialistas fizeram.

    * cada paciente possui 61 B-scans, mas nem todos possuem anotação
    * inicialmente está sendo analisadas as anotações da chave "manualLayers1"

# 2 - preparação de imagens
Por meio dos arquivos .mat foram filtradas as imagens de cada paciente que possuem anotação por meio
de uma classe definida em que após localizar essas imagens irá definir :

    * images - é criado um diretório para armazenar as imagens que possuem anotações já cortadas
                            |
              |__manualLayers1 - imagens usadas para criar mascaras de acordo com o especialista 1
              
              |
              |__manualLayers2 - imagens usadas para criar mascaras de acordo com o especialista 2
              
              |
              |__manualFluid1 -  imagens usadas para criar mascaras dos fluidos de acordo com o especialista 1
              
              |
              |__manualFluid2 - imagens usadas para criar mascaras dos fluidos de acordo com o especialista 2
    
    * masks 
              |
              |__manualLayers1 - máscaras das camadas definidas pelo especialista 1
              
              |
              |__manualLayers2 - máscaras das camadas definidas pelo especialista 2
              
              |
              |__manualFluid1 -  máscaras dos flúidos definidas pelo especialista 1
              
              |
              |__manualFluid2 - máscaras dos flúidos definidas pelo especialista 2

Uma mesma imagem possui as 4 anotações menciondas, ou seja, não há diferentes imagens para cada
tipo de máscara mencionado.


# 3 - treinamento da rede neural

# info
Esse este estudo se baseia no artigo https://opg.optica.org/boe/fulltext.cfm?uri=boe-6-4-1172&id=312754 que busca identificar em imagens de  OCT ÁREAS DE FLUÍDOS e SETE CAMADAS DA RETINA COM
DME (Diabetic Macular Edema). Esse artigo propõe um algoritmo que busca prever a posição dessas regiões nas imagens. Fizeram isso em 110 B-scans de 10 pacientes com DME severa com um ALGORITMO BASEADO EM REGRESSÃO DE KERNEL

Layers
        |
        |__ILM : Inner Limiting Membrane
        |
        |__NFL/GCl : Nerve Fiber Layer 
        |
        |__IPL/INL : Inner Plexiform layer e Inner Nuclear Layer
        |
        |__OPL/ONL : Outer Plexiform Layer e Outer Nuclear Layer
        |
        |__ISM/ISE : Inner Segment Myeloid e Inner Segment Ellipsoid
        |
        |__OS/RPE : Outer Segment e Retinal Pigment Epithelium
        |
        |__BM : Bruch's Membrane

Dois oftalmologistas segmentaram manualmente todas as regiões cheias de líquido e oito limites da camada retinal usando software personalizado (DOCTRAP V50.9) para 110 imagens (11 por paciente).

=============================================================================================
# Structured layer surface segmentation for retina OCT using fully convolutional regression networks

camadas da retina são importantes biomarcadores para anomalias na retina como edema macular diabético (DME)
ou anomalias neurológicas como Esclerose Múltipla (EM). E a segmentação manual dessas imagens de OCT são de-
moradas.


- Roy et al. (2017) usaram uma rede totalmente convolucional (FCN) para rotular cada pixel em oito classes de camadas, edema e fundo

- Lee et al. (2017a) e Schlegl et al. (2018) cada um usou um FCN para segmentar edema e fluidos retinianos

- Venhuizen et al. (2017) usado um FCN para segmentar toda a retina sem segmentar cada camada

- O segundo tipo de classificador rotula pixels como superfícies (ou seja,limites entre as camadas da 
retina ou entre a retina e seus
antecedentes)

retinal layers:

    |_the retinal nerve fiber layer (RNFL);
    |
    |_the ganglion cell layer (GCL) combined with the inner plexiform layer (IPL), denoted as GCIP;
    |
    |_the inner nuclear layer (INL); the outer plexiform layer (OPL);
    |
    |_the outer nuclear layer (ONL), the inner segment (IS);
    |
    |_the outer segment (OS);
    |
    |_and the retinal pigment epithelium (RPE);
    |
    |_Surfaces between these layers are identified by hyphenating
    |
    |_their acronyms. The other named surfaces are: the inner limiting membrane (ILM);
    |
    |_the external limiting membrane (ELM); --> não tem no dataset da IC
    |
    |_and Bruch’s Membrane (BM).
    |
    |_Finally, above the RNFL is the vitreous and below the BM is the choroid.

Nossa rede tem o benefício de: 
1) ser treinado de ponta a ponta;

2) melhorar a precisão contra o estado da arte;

3) ser leve porque não usa uma camada totalmente conectada para regressão. Também realizamos uma análise da inclinação da superfície para mostrar que a conectividade da superfície está bem restrita, mesmo sem restrições explícitas, como no caso métodos gráficos.

A rede construída tem dois ramos de saída :

    * o primeiro gera uma segmentação de camadas com rotulagem em pixels e lesões

    * o segundo modelo a distribuição das posições da superfície e produz as posições
      de cada superfície em cada coluna (A-scan)

    Ambos os ramos compartilham o mesmo extrator de características que é U-Net residual.
    O input é uma imagem de tres canais

======================================
Método de avaliação do modelo proposto
======================================

O esquema de rotulagem por pixel tem problemas topológicos com camadas biologicamente estruturadas. No entanto, lesões (líquido ou edema) que aparecem em locais diferentes e
com formas diferentes não podem receber uma topologia fixa e, portanto, a rotulagem em 
pixels é apropriada.

usaram dois datasets: t (He et al., 2019c) contains 14 healthy controls (HC) and 21 people with MS (PwMS) e o (Chiu et al., 2015) contains 110 Bscans (496 × 768) from 10 diabetic macular edema (DME) patients.

No segundo dataset (usado na IC), os 5 primeiros pacientes possuem DME severa com sequelas
na estrutura retiniana. Foram delineadas 8 camadas e os edemas encontrados. A divisão foi
50% - 50% treino/teste segundo (Chiu et al., 2015; Rathke et al., 2017; Karri et al., 2016; Roy et al., 2017). Treinaram nas 55 últimas imagens até o treinamento convergir e foi testada
nas primeiras 55 imagens que apresentam edema gravíssimo.

- as imagens foram achatadas e cortadas para 224 x 768 e usaram pytorch
- hiperparâmetros: Adam optmizer; learning rate: 10e-4; weight decay: 10e-4; minibatch size: 2
- aplicou-se data augmentation com horizontal flipping e vertical scaling, ambos com probabilidade 0.5, e a imagem em escala foi cortada no mesmo tamanho anttes de dimensionar

No DME Datasets fizeram uma comparação de resultados. Mediram o Dice Score de
Chiu et al. (2015), ReLayNet e o modelo proposto que alcançaram os valores 0.56,
0.70 e 0.70 respectivamente

==============================================================================================
# OCT layers segmetation using U-NET semantic segmentation and RESNET34 encoder-decoder

accuracy e mean Intersection over Union (mIoU) foram métricas utilizadas levando em consideração dois benchmarks: DeepLabV3Plus e UnetP++. Foi usado Pytorch e python 3.9

* Wenzhou Medical University (WMU) forneceu o dataset

encontrou-se 0.92, 0.90 e 0.95 de acurácia, sensibilidade e especificidade respectivamente.

600 imagens de OCT que procedem de 40 olhos e 15 imagens selecionadas para o primeiro dataset

mIoU = pontuação média da IoU em todas as classes ou conjunto de dados


==============================================================================================
A avaliação de modelos de segmentação semântica deve levar em consideração se os dados são balanceados ou não, pois a irregularidade de quantidades de pixels para cada classe pode
mascarar os resultados, o que leva a métrica de acurácia não ser muito eficiente na avaliação

* DICE LOSS

    essa métrica avalia a dissimilaridade entre a segmentação predita e a segmentação 
    verdadeira. Isso é uma variação de Dice Similarity Coefficient conhecido como
    Sørensen–Dice coefficient, métrica que compara a similaridade entre dois samples.
    A Dice Loss compara a similaridade de duas classificações binárias da segmentação
    verdadeira e da segmentação predita. O objetivo é diminuir a diferença entre as
    segmentações, a qual é chamada de função de perda.

        DiceLoss(𝑦,𝑝̅ ) = 1−(2𝑦𝑝̅ +1)/(𝑦+𝑝̅+1)

        Sørensen–Dice coefficient(𝑦,𝑝̅) = 2*(𝑦 ∩ 𝑝̅)/(𝑦 + 𝑝̅)

        Dice Loss = 1 - Dice Coefficient

        𝑦 - segmentação verdadeira
        𝑝̅ - segmentação predita

    ela gera um valor entre 0 e 1, sendo 0 dissimilaridade e 1 similaridade completa entre
    as segmentação. É o oposto do Dice Similarity Coefficient.


*******************************************************************************************
* smooth = 10e-3                                                                          *
* def dice_coef(y_true, y_pred):                                                          *
*    y_true_f = K.flatten(y_true)                                                         *
*    y_pred_f = K.flatten(y_pred)                                                         *
*    intersection = K.sum(y_true_f * y_pred_f)                                            *
*    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)   *
*******************************************************************************************


* IoU (INTERSECTION OVER UNION - JACCARD INDEX) 

    IoU = (𝑦 ∩ 𝑝̅)/(𝑦 ∪ 𝑝̅)
    IoU Loss = 1 - Iou

**********************************************************************************************
* def mIOU(prediction, label, num_classes):                                                  *
*    prediction= prediction.max(1)[1].float().cpu().numpy()                                  *
*    label = label.float().cpu().numpy()                                                     *
*    iou_list = list()                                                                       *
*    present_iou_list = list()                                                               *
*                                                                                            *
*    for sem_class in range(num_classes):                                                    *
*        pred_inds = (pred == sem_class)                                                     *
*        target_inds = (label == sem_class)                                                  *
*        if target_inds.sum().item() == 0:                                                   *
*            iou_now = float('nan')                                                          *
*        else:                                                                               *
*            intersection_now = (pred_inds[target_inds]).sum().item()                        *
*            union_now = pred_inds.sum().item() + target_inds.sum().item() - intersection_now*
*            iou_now = float(intersection_now) / float(union_now)                            *
*            present_iou_list.append(iou_now)                                                *
*        iou_list.append(iou_now)                                                            *
*    miou = np.mean(present_iou_list)                                                        *
*    return miou                                                                             *
**********************************************************************************************n