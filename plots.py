import open3d as o3d
import numpy as np
import trimesh
from trimesh.path.polygons import second_moments
from shapely.geometry import box, MultiPolygon
import matplotlib.pyplot as plt
import os
import json
import multiprocessing
from functools import partial
import time  # Importado para o timer
from functools import wraps # Importado para o timer

def timer(func):
    """Um decorador simples para medir o tempo de execução da função."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"\n[TIMING] Função '{func.__name__}' levou {elapsed_time:.4f} segundos.\n")
        return result
    return wrapper

# --- PARÂMETROS PRINCIPAIS QUE VOCÊ PODE AJUSTAR ---
ARQUIVO_MALHA_CASCO = "casco_bulk_carrier.stl" # Coloque o nome do seu .stl, .obj, ou .fbx aqui
NIVEL_AGUA_Z = 8
DENSIDADE_AGUA = 1025.0  # kg/m³ (Água salgada)
CM_POR_M = 100.0         # Fator de conversão (100 cm por metro)
# ---------------------------------------------------

@timer
def carregar_malha_do_arquivo(file_path):
    """
    Carrega um arquivo de malha (STL, OBJ, FBX) diretamente.
    Esta função substitui 'carregar_nuvem_de_pontos' e 'gerar_casco_poisson'.
    """
    print(f"Carregando malha do arquivo: {file_path}")
    if not os.path.exists(file_path):
        print(f"ERRO: Arquivo não encontrado: {file_path}")
        return None
    
    try:
        # Open3D lê diversos formatos
        casco_malha = o3d.io.read_triangle_mesh(file_path)
    except Exception as e:
        print(f"ERRO ao ler o arquivo de malha: {e}")
        return None

    if not casco_malha.has_vertices():
        print("ERRO: O arquivo foi lido, mas a malha está vazia (sem vértices).")
        return None

    print(f"Malha carregada: {len(casco_malha.vertices)} vértices, {len(casco_malha.triangles)} triângulos.")
    
    # É uma boa prática pré-processar a malha
    casco_malha.remove_duplicated_vertices()
    casco_malha.remove_unreferenced_vertices()
    
    print("Malha do casco carregada e limpa.")
    return casco_malha

@timer
def visualizar_casco_apenas(casco_malha):
    """
    Mostra apenas a malha do casco gerada para inspeção visual.
    O script pausa aqui até você fechar a janela.
    """
    print("\n--- PASSO DE DEPURAÇÃO ---")
    print("Mostrando a malha do casco gerada pelo Poisson.")
    print("Inspecione por buracos ou faces invertidas.")
    print("FECHE A JANELA para continuar o script e tentar calcular o volume...")
    
    casco_malha.paint_uniform_color([0.7, 0.7, 0.7]) # Cor cinza
    
    # Adiciona eixos para referência
    coordenadas = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    
    o3d.visualization.draw(
        [casco_malha, coordenadas],
        show_skybox=True,
        bg_color=[0.9, 0.9, 0.9, 1.0]
    )
    print("Continuando script...")
    print("-------------------------\n")

def visualizar_secao_mestra(secao_2d, caixa_corte_2d, poligonos_submersos):
    """
    Usa o Matplotlib para plotar a seção mestra (muito mais robusto que o viewer 2D do trimesh).
    """
    print("\n--- VISUALIZAÇÃO 2D COM MATPLOTLIB ---")
    print("Abrindo visualizador Matplotlib...")
    print("  (Preto = Seção Mestra Total, Azul = Caixa de Corte, Vermelho = Área Submersa)")

    try:
        fig, ax = plt.subplots()
        
        # 1. Plotar Seção Mestra Total (Preto)
        # secao_2d.polygons_full é uma lista de polígonos Shapely
        if secao_2d and secao_2d.polygons_full:
            print("  Plotando 'secao_2d' (Preto)...")
            for poly in secao_2d.polygons_full:
                if poly.is_empty: continue
                # Plota o contorno externo
                ax.plot(*poly.exterior.xy, color='black', alpha=0.7, linewidth=1.5)
                # Plota os buracos (contornos internos)
                for interior in poly.interiors:
                     ax.plot(*interior.xy, color='black', alpha=0.7, linewidth=1)

        # 2. Plotar Caixa de Corte (Azul)
        if caixa_corte_2d is not None and not caixa_corte_2d.is_empty:
            print("  Plotando 'caixa_corte_2d' (Azul)...")
            ax.plot(*caixa_corte_2d.exterior.xy, color='blue', alpha=0.5, linestyle='--')

        # 3. Plotar Polígonos Submersos (Vermelho)
        # poligonos_submersos é a lista 'poligonos_submersos_am'
        if poligonos_submersos:
            print("  Plotando 'poligonos_submersos' (Vermelho)...")
            for poly in poligonos_submersos:
                if poly.is_empty: continue
                # Preenche a área submersa
                ax.fill(*poly.exterior.xy, color='red', alpha=0.5)
                # Preenche os buracos (interiores) do submerso com branco
                for interior in poly.interiors:
                     ax.fill(*interior.xy, color='white', alpha=1.0)

        ax.set_aspect('equal', 'box')
        ax.set_title("Visualização da Seção Mestra")
        # Lembre-se: No plano YZ, o Calado (Z) está no eixo Y do plot
        ax.set_xlabel("Eixo Y (m)")
        ax.set_ylabel("Eixo Z (Calado) (m)")
        plt.grid(True)
        print("  Mostrando plot...")
        plt.show() # Esta chamada abre a janela
        print("  Visualização Matplotlib fechada.")

    except Exception as e:
        print(f"  [ERRO AO PLOTAR COM MATPLOTLIB]: {e}")
        print("  A visualização 2D falhou, mas os cálculos continuarão.")
        
    print("--- FIM VISUALIZAÇÃO 2D ---")
    print("  Debug: Retornando ao script principal.")

def gerar_bloco_agua(casco_trimesh, water_level):
    """
    Cria um "bloco de água" (uma caixa) que será usado para a interseção.
    Usa Trimesh para criar a caixa de forma robusta.
    """
    print(f"Passo 2: Gerando bloco de água (Plano Z = {water_level})...")
    
    # Pega os limites do casco para fazer um bloco de água maior
    # 'casco_trimesh' (Trimesh) usa .bounds em vez de .get_axis_aligned_bounding_box()
    min_bound = casco_trimesh.bounds[0]
    max_bound = casco_trimesh.bounds[1]

    # Adiciona um "padding" em X e Y para garantir um corte limpo
    padding = 2.0 
    
    # Define os limites da caixa de água
    bounds_agua = [
        [min_bound[0] - padding, min_bound[1] - padding, min_bound[2] - padding], # Canto inferior
        [max_bound[0] + padding, max_bound[1] + padding, water_level]             # Canto superior (na linha d'água)
    ]
    
    # Cria a caixa (bloco de água)
    bloco_agua_trimesh = trimesh.primitives.Box(bounds=bounds_agua)
    
    return bloco_agua_trimesh

@timer
def calcular_todas_as_propriedades_hidrostaticas(casco_trimesh, water_level):
    
    print("Iniciando cálculos hidrostáticos...")
    
    # -----------------------------------------------------
    # PASSO 1: CALCULAR VOLUME E CENTROIDE (Interseção, VCB, LCB, TCB)
    # -----------------------------------------------------
    
    bloco_agua_trimesh = gerar_bloco_agua(casco_trimesh, water_level)
    
    try:
        malha_submersa = casco_trimesh.intersection(bloco_agua_trimesh)
    except Exception as e:
        print(f"ERRO CRÍTICO NA INTERSEÇÃO: {e}")
        return None 

    if malha_submersa.is_empty:
        print("ERRO: Malha submersa está vazia. O nível da água pode estar abaixo do casco.")
        return None 
    
    volume = malha_submersa.volume
    centroide = malha_submersa.center_mass
    
    # Parâmetros Auxiliares
    calado = water_level
    vcb = centroide[2]
    lcb = centroide[0]
    tcb = centroide[1]
    
    # -----------------------------------------------------
    # PASSO 2: CALCULAR PROPRIEDADES DO PLANO DE FLUTUAÇÃO (AWP, IT, IL, LWL, BWL, LCF)
    # -----------------------------------------------------
    print("Calculando propriedades do plano de flutuação...")
    
    fatia_3d_path = casco_trimesh.section(plane_origin=[0, 0, water_level], plane_normal=[0, 0, 1])
    if fatia_3d_path is None or fatia_3d_path.is_empty:
         print(f"Aviso: Corte no nível d'água (Z={water_level}) não retornou geometria.")
         return None

    # 1. Criamos nossa própria transformação para o plano XY na origem (0,0)
    transform_xy = trimesh.geometry.plane_transform(
        origin=[0, 0, water_level], normal=[0, 0, 1]
    )
         
    # 2. Forçamos o to_2D() a usar essa transformação
    # A geometria 2D resultante estará no sistema de coordenadas global (X=X, Y=Y)
    fatia_2d_path, _ = fatia_3d_path.to_2D(to_2D=transform_xy)

    total_area = 0.0
    total_Ixx_origem = 0.0 # Momento de Inércia sobre a origem (X=0)
    total_Iyy_origem = 0.0 # Momento de Inércia sobre a origem (Y=0)
    total_moment_x = 0.0 # Momento de área para LCF
    total_moment_y = 0.0 # Momento de área para TCF

    if not fatia_2d_path.polygons_full:
        print("Aviso: O corte 2D não gerou polígonos fechados.")
        return None

    # Itera sobre os polígonos 2D do plano de flutuação
    for polygon in fatia_2d_path.polygons_full:
        area = polygon.area
        centroid_2d = polygon.centroid
        i = fatia_2d_path.polygons_full.index(polygon) + 1

        try:
            moments_origem = second_moments(polygon, return_centered=False)
            total_area += area
            total_Ixx_origem += moments_origem[0]  # Momento sobre X=0
            total_Iyy_origem += moments_origem[1]  # Momento sobre Y=0
            
            total_moment_x += centroid_2d.x * area
            total_moment_y += centroid_2d.y * area
        except Exception as e:
            print(f"  DEBUG: Falha ao calcular momentos do polígono {i}: {e}")
            continue
    
    if total_area < 1e-6:
        print("Aviso: Área do plano de flutuação é zero.")
        return None

    # Área do plano de flutuação
    area_plano_flutuacao = total_area
    
    # LCF e TCF (Centroides globais do plano de flutuação)
    lcf = total_moment_x / total_area
    tcf = total_moment_y / total_area

    # I = I_origem - A*d^2
    # Queremos os momentos sobre o LCF (d=lcf) e o TCF (d=tcf)
    momento_inercia_transversal = total_Ixx_origem - total_area * (tcf**2) # I_T (sobre o eixo Y)
    momento_inercia_longitudinal = total_Iyy_origem - total_area * (lcf**2) # I_L (sobre o eixo X)
    
    # Dimensões da Linha d'Água (LWL e BWL)
    bounds_2d = fatia_2d_path.bounds
    LWL = bounds_2d[1, 0] - bounds_2d[0, 0] 
    BWL = bounds_2d[1, 1] - bounds_2d[0, 1]

    # -----------------------------------------------------
    # PASSO 3: CALCULAR Am (Área da Seção Mestra)
    # -----------------------------------------------------
    print("Calculando Área da Seção Mestra (Am)...")
    Am = 0.0
    # A seção mestra é tipicamente na metade do LWL
    x_mestra = bounds_2d[0, 0] + LWL / 2.0 
    secao_mestra_3d = casco_trimesh.section(plane_origin=[x_mestra, 0, 0], plane_normal=[1, 0, 0])

    if LWL > 1e-6:
        # 1. Define uma nova transformação para o plano YZ
        transform_yz = trimesh.geometry.plane_transform(
            origin=[x_mestra, 0, 0], normal=[1, 0, 0]
        )
        
        if secao_mestra_3d is not None and not secao_mestra_3d.is_empty:
            # Converte a fatia 3D (X-normal) para 2D (YZ-plane)
            secao_2d, _ = secao_mestra_3d.to_2D(to_2D=transform_yz)
            
            # No plano YZ, o calado (Z) está agora no eixo Y (índice 1)
            # Como forçamos a origem Z=0, a linha d'água está em Y = water_level
            
            caixa_corte_2d = box(
                minx=-water_level,  # Controla o eixo X do gráfico (que é o Calado Z)
                maxx=calado*2,      # Controla o eixo X do gráfico (que é o Calado Z)
                miny=-BWL*1.5,      # Controla o eixo Y do gráfico (que é a Boca Y)
                maxy=BWL*1.5        # Controla o eixo Y do gráfico (que é a Boca Y)
            )
                    
        poligonos_submersos_am = []
        Am = 0.0 # Reinicia Am para recalcular
        for poly in secao_2d.polygons_full:
            poly_submerso = poly.intersection(caixa_corte_2d)
            if not poly_submerso.is_empty:
                # Lida com MultiPolygons ou GeometryCollections
                if hasattr(poly_submerso, 'geoms'): 
                    for geom in poly_submerso.geoms:
                        # Garante que é um Polígono e tem área
                        if geom.geom_type == 'Polygon' and geom.area > 1e-6:
                             poligonos_submersos_am.append(geom)
                             Am += geom.area
                # Lida com Polígonos simples
                elif poly_submerso.geom_type == 'Polygon' and poly_submerso.area > 1e-6:
                    poligonos_submersos_am.append(poly_submerso)
                    Am += poly_submerso.area

    # -----------------------------------------------------
    # PASSO 4: APLICAR FÓRMULAS HIDROSTÁTICAS
    # -----------------------------------------------------
    print("Calculando coeficientes e resultados finais...")
    
    # Deslocamento
    deslocamento = volume * DENSIDADE_AGUA / 1000.0 
    
    # Estabilidade Transversal
    bmt = momento_inercia_transversal / volume
    kmt = vcb + bmt

    # Estabilidade Longitudinal
    bml = momento_inercia_longitudinal / volume
    kml = vcb + bml

    # Outras Propriedades
    tpc = (area_plano_flutuacao * DENSIDADE_AGUA) / CM_POR_M
    mtc = (momento_inercia_longitudinal * DENSIDADE_AGUA) / (CM_POR_M * LWL) if LWL > 1e-6 else 0.0

    # Coeficientes de Forma
    denominador_bloco = LWL * BWL * calado
    cb = volume / denominador_bloco if denominador_bloco > 1e-6 else 0.0

    denominador_prismatico = Am * LWL
    cp = volume / denominador_prismatico if denominador_prismatico > 1e-6 else 0.0
    
    denominador_plano_flutuacao = LWL * BWL
    cwp = area_plano_flutuacao / denominador_plano_flutuacao if denominador_plano_flutuacao > 1e-6 else 0.0

    # Cm = Am / (BWL * T)
    cm = Am / (BWL * calado) if BWL * calado > 1e-6 else 0.0
    
    # -----------------------------------------------------
    # PASSO 5: Retornar Dicionário
    # -----------------------------------------------------
    
    resultados = {
        'Calado (m)': calado,
        'Volume (m³)': volume, 
        'Desloc. (t)': deslocamento,
        'AWP (m²)': area_plano_flutuacao, 
        'LWL (m)': LWL, 
        'BWL (m)': BWL,
        'LCB (m)': lcb, 
        'VCB (m)': vcb, 
        'TCB (m)': tcb, 
        'LCF (m)': lcf,
        'BMt (m)': bmt, 
        'KMt (m)': kmt, 
        'BMl (m)': bml, 
        'KMl (m)': kml,
        'TPC (t/cm)': tpc, 
        'MTc (t·m/cm)': mtc, 
        'Cb': cb, 
        'Cp': cp,
        'Cwp': cwp, 
        'Cm': cm,
        'casco_trimesh_reparado': casco_trimesh, 
        'bloco_agua_trimesh': bloco_agua_trimesh
    }
    
    return resultados

# --- FUNÇÃO DE IMPRESSÃO ---
def imprimir_resultados(props):
    print("\n--- RESULTADO HIDROSTÁTICO COMPLETO ---")
    print("--------------------------------------------------")
    print(f"Nível da Água (Z):           {props['Calado (m)']:.4f} m")
    print(f"Volume Submerso (∇):         {props['Volume (m³)']:.4f} m³")
    print(f"Deslocamento (t):            {props['Desloc. (t)']:.4f} t")
    print("--------------------------------------------------")
    print(f"Centro de Carena (LCB, TCB, VCB): (X={props['LCB (m)']:.4f}, Y={props['TCB (m)']:.4f}, Z={props['VCB (m)']:.4f}) m")
    print(f"Centro de Flutuação (LCF):   {props['LCF (m)']:.4f} m")
    print(f"Área Plano Flutu. (AWP):     {props['AWP (m²)']:.4f} m²")
    print("--------------------------------------------------")
    print(f"Raio Metacêntrico Transv. (BMt): {props['BMt (m)']:.4f} m")
    print(f"Altura Metacêntrica Transv. (KMt): {props['KMt (m)']:.4f} m")
    print(f"Raio Metacêntrico Long. (BMl): {props['BMl (m)']:.4f} m")
    print(f"Altura Metacêntrica Long. (KMl): {props['KMl (m)']:.4f} m")
    print("--------------------------------------------------")
    print(f"LWL (m) / BWL (m):           {props['LWL (m)']:.4f} / {props['BWL (m)']:.4f} m")
    print(f"Coef. Bloco (Cb):            {props['Cb']:.4f}")
    print(f"Coef. Prismático (Cp):       {props['Cp']:.4f}")
    print(f"Coef. Plano Flutu. (Cwp):    {props['Cwp']:.4f}")
    print(f"Coef. Seção Mestra (Cm):     {props['Cm']:.4f}")
    print("--------------------------------------------------")
    print(f"TPC (t/cm):                  {props['TPC (t/cm)']:.4f}")
    print(f"MTc (t·m/cm):                {props['MTc (t·m/cm)']:.4f}")
    print("--------------------------------------------------\n")

# --- FUNÇÃO PRINCIPAL ---
# @timer
# def main():
#     # Passo 1: Carregar a MALHA
#     casco_malha = carregar_malha_do_arquivo(ARQUIVO_MALHA_CASCO)
#     if casco_malha is None:
#         return

#     CALADO_MINIMO = 1.0
#     CALADO_MAXIMO = NIVEL_AGUA_Z  # Usa o valor 8.0 definido no topo
#     PASSO_CALADO = 0.5  # Calcular a cada 0.5m

#     calados_para_calcular = np.arange(CALADO_MINIMO, CALADO_MAXIMO + PASSO_CALADO, PASSO_CALADO)

#     print(f"\nIniciando loop de cálculos para {len(calados_para_calcular)} calados (de {CALADO_MINIMO}m a {CALADO_MAXIMO}m)...\n")

#     # Lista principal para armazenar todos os resultados
#     curvas_hidrostaticas = []

#     for calado_atual in calados_para_calcular:
#         print(f"--- Calculando para o calado: {calado_atual:.2f} m ---")
        
#         # Passo 2: Calcular TUDO para o calado atual
#         # A função 'calcular_...' já retorna um dicionário
#         props_hidrostaticas = calcular_todas_as_propriedades_hidrostaticas(casco_malha, calado_atual)

#         if props_hidrostaticas is None:
#             print(f"Cálculo falhou para o calado {calado_atual:.2f} m. Pulando.")
#             continue

#         # (Opcional) Imprime os resultados no console para feedback
#         # imprimir_resultados(props_hidrostaticas)
        
        
#         # 1. Limpar o dicionário 'props' dos objetos grandes
#         #    (Não podemos salvar malhas Trimesh em um JSON)
#         if 'casco_trimesh_reparado' in props_hidrostaticas:
#             del props_hidrostaticas['casco_trimesh_reparado']
#         if 'bloco_agua_trimesh' in props_hidrostaticas:
#             del props_hidrostaticas['bloco_agua_trimesh']
        
#         # 2. Criar o dicionário no formato exato que você pediu
#         resultado_formatado = {
#             "calado": calado_atual,
#             "dados_hidrostaticos": props_hidrostaticas # 'props' agora só contém os valores
#         }
        
#         # 3. Adicionar à lista principal
#         curvas_hidrostaticas.append(resultado_formatado)
#         print(f"--- Resultados para {calado_atual:.2f} m salvos na lista. --- \n")
    
#     print(f"\nCálculo em loop concluído. {len(curvas_hidrostaticas)} conjuntos de dados gerados.")
    
#     # Passo 4: Salvar a lista completa em um arquivo JSON
#     NOME_ARQUIVO_SAIDA = "resultados_hidrostaticos_completos.json"
#     try:
#         with open(NOME_ARQUIVO_SAIDA, 'w', encoding='utf-8') as f:
#             # 'default=str' converte tipos numpy (como np.float64) para float padrão do JSON
#             json.dump(curvas_hidrostaticas, f, indent=4, default=str)
#         print(f"Resultados completos salvos em: {NOME_ARQUIVO_SAIDA}")
#     except Exception as e:
#         print(f"ERRO AO SALVAR JSON: {e}")

#     # (Opcional) Imprimir o primeiro resultado da lista para verificação
#     if curvas_hidrostaticas:
#          print(f"\nExemplo do ultimo resultado salvo (calado={NIVEL_AGUA_Z:.2f}m):")
#          print(json.dumps(curvas_hidrostaticas[-1], indent=2, default=str))

# --- FUNÇÃO PRINCIPAL ---

# Precisamos de uma função "wrapper" que possa ser usada pelo pool de multiprocessing.
# Ela só pode ter UM argumento (o calado), pois o pool.map funciona assim.
# A malha (casco_trimesh) será "pré-preenchida" usando functools.partial.
def calcular_para_um_calado(calado_atual, casco_trimesh_reparado):
    """
    Função wrapper para o multiprocessing. Recebe o calado e a malha já reparada.
    """
    print(f"--- Iniciando cálculo para o calado: {calado_atual:.2f} m ---")
    
    props = calcular_todas_as_propriedades_hidrostaticas(casco_trimesh_reparado, calado_atual)

    if props is None:
        print(f"Cálculo falhou para o calado {calado_atual:.2f} m. Retornando None.")
        return None

    # --- LÓGICA DE SALVAMENTO (agora dentro do worker) ---
    
    # 1. Limpar o dicionário 'props' dos objetos grandes
    if 'casco_trimesh_reparado' in props:
        del props['casco_trimesh_reparado']
    if 'bloco_agua_trimesh' in props:
        del props['bloco_agua_trimesh']
    
    # 2. Criar o dicionário no formato final
    resultado_formatado = {
        "calado": calado_atual,
        "dados_hidrostaticos": props
    }
    
    print(f"--- Cálculo para {calado_atual:.2f} m CONCLUÍDO. ---")
    return resultado_formatado

@timer
def main():
    # Passo 1: Carregar a MALHA (Open3D)
    casco_malha = carregar_malha_do_arquivo(ARQUIVO_MALHA_CASCO)
    if casco_malha is None:
        return

    # --- OTIMIZAÇÃO SERIAL: Reparar a malha UMA VEZ ---
    print("\nConvertendo e reparando a malha (Trimesh) uma única vez...")
    casco_trimesh_reparado = trimesh.Trimesh(
        vertices=np.asarray(casco_malha.vertices), faces=np.asarray(casco_malha.triangles)
    )
    casco_trimesh_reparado.process()
    casco_trimesh_reparado.fill_holes()
    casco_trimesh_reparado.fix_normals()
    print("Malha reparada e pronta para o processamento paralelo.\n")
    # --- Fim da Otimização Serial ---

    # --- PARÂMETROS DO LOOP ---
    CALADO_MINIMO = 1.0
    CALADO_MAXIMO = NIVEL_AGUA_Z
    PASSO_CALADO = 0.5
    calados_para_calcular = np.arange(CALADO_MINIMO, CALADO_MAXIMO + PASSO_CALADO, PASSO_CALADO)
    
    print(f"Iniciando processamento paralelo para {len(calados_para_calcular)} calados...")
    
    # --- OTIMIZAÇÃO PARALELA (Multiprocessing) ---
    
    # "Pré-preenche" a função 'calcular_para_um_calado' com o argumento 'casco_trimesh_reparado'.
    # A função resultante agora só precisa de um argumento: 'calado_atual'.
    funcao_worker = partial(calcular_para_um_calado, 
                            casco_trimesh_reparado=casco_trimesh_reparado)
    
    # Inicia um "pool" de workers. Usará todos os núcleos disponíveis.
    with multiprocessing.Pool() as pool:
        # pool.map aplica a 'funcao_worker' a cada item da lista 'calados_para_calcular'
        # e distribui o trabalho por todos os seus núcleos de CPU.
        # 'resultados_brutos' será uma lista de dicionários (ou None se falhou)
        resultados_brutos = pool.map(funcao_worker, calados_para_calcular)
    
    # --- FIM DO PARALELISMO ---

    # Limpar a lista final de resultados que falharam (None)
    curvas_hidrostaticas = [r for r in resultados_brutos if r is not None]

    print(f"\nCálculo em loop concluído. {len(curvas_hidrostaticas)} conjuntos de dados gerados.")
    
    # Passo 4: Salvar a lista completa em um arquivo JSON
    NOME_ARQUIVO_SAIDA = "resultados_hidrostaticos_completos.json"
    try:
        with open(NOME_ARQUIVO_SAIDA, 'w', encoding='utf-8') as f:
            json.dump(curvas_hidrostaticas, f, indent=4, default=str)
        print(f"Resultados completos salvos em: {NOME_ARQUIVO_SAIDA}")
    except Exception as e:
        print(f"ERRO AO SALVAR JSON: {e}")

    # (Opcional) Imprimir o último resultado da lista para verificação
    if curvas_hidrostaticas:
         print(f"\nExemplo do último resultado salvo (calado={curvas_hidrostaticas[-1]['calado']}m):")
         print(json.dumps(curvas_hidrostaticas[-1], indent=2, default=str))


if __name__ == "__main__":
    main()