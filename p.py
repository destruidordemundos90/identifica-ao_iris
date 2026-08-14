import cv2
import numpy as np
import sqlite3
import sys

# --- 1. BANCO DE DADOS (SALVA A TEXTURA DA ÍRIS) ---
def iniciar_banco():
    conexao = sqlite3.connect("empresa_seguranca.db")
    cursor = conexao.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS funcionarios (
            cpf TEXT PRIMARY KEY,
            nome TEXT NOT NULL,
            iris_textura BLOB NOT NULL
        )
    ''')
    conexao.commit()
    conexao.close()

def cadastrar_funcionario(nome, cpf, matriz_caracteristicas):
    try:
        conexao = sqlite3.connect("empresa_seguranca.db")
        cursor = conexao.cursor()
        
        # Converte os pontos da íris em bytes puros
        dados_bytes = matriz_caracteristicas.tobytes()
        
        cursor.execute("INSERT INTO funcionarios (cpf, nome, iris_textura) VALUES (?, ?, ?)", 
                       (cpf, nome, dados_bytes))
        conexao.commit()
        print(f"\n[SUCESSO] {nome} cadastrado com sucesso!")
    except sqlite3.IntegrityError:
        print("\n[ERRO] Este CPF já está cadastrado no sistema!")
    finally:
        conexao.close()

def buscar_iris(matriz_atual):
    if matriz_atual is None or len(matriz_atual) == 0:
        return None

    conexao = sqlite3.connect("empresa_seguranca.db")
    cursor = conexao.cursor()
    cursor.execute("SELECT nome, iris_textura FROM funcionarios")
    usuarios = cursor.fetchall()
    conexao.close()

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for nome, textura_bytes in usuarios:
        try:
            textura_salva = np.frombuffer(textura_bytes, dtype=np.uint8).reshape(-1, 32)
            combinacoes = bf.match(matriz_atual, textura_salva)
            combinacoes_boas = [m for m in combinacoes if m.distance < 30]
            
            # Se encontrar mais de 10 pontos parecidos, valida o funcionário
            if len(combinacoes_boas) > 10:
                return nome
        except Exception:
            continue
    return None 

# --- 2. EXTRATOR DE DETALHES DA ÍRIS ---
def extrair_textura_iris(imagem_cinza, x, y, raio):
    """Recorta o olho detectado e extrai os microdetalhes matemáticos."""
    h, w = imagem_cinza.shape
    
    y1, y2 = max(0, int(y - raio)), min(h, int(y + raio))
    x1, x2 = max(0, int(x - raio)), min(w, int(x + raio))
    
    regiao_iris = imagem_cinza[y1:y2, x1:x2]
    if regiao_iris.size == 0:
        return None
        
    regiao_iris = cv2.resize(regiao_iris, (100, 100))
    
    orb = cv2.ORB_create(nfeatures=200)
    pontos_chave, descritores = orb.detectAndCompute(regiao_iris, None)
    
    return descritores

# --- 3. FLUXO PRINCIPAL ---
iniciar_banco()

print("--- SISTEMA DE BIOMETRIA DE ÍRIS REAL ---")
print("1. Cadastrar Novo Funcionário")
print("2. Iniciar Modo de Desbloqueio (Portaria)")
opcao = input("Escolha a opção (1 ou 2): ")

webcam = cv2.VideoCapture(0)

# Garante resolução padrão rápida
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if opcao == "1":
    nome = input("Digite o nome do funcionário: ")
    cpf = input("Digite o CPF (Apenas números): ")
    print("\nAproxime o olho da câmera e pressione 'S' para salvar...")
    
    while True:
        sucesso, frame = webcam.read()
        if not sucesso: 
            break
        
        imagem_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        imagem_borrada = cv2.medianBlur(imagem_cinza, 5)
        circulos = cv2.HoughCircles(imagem_borrada, cv2.HOUGH_GRADIENT, 1, 100, 
                                    param1=50, param2=30, minRadius=25, maxRadius=70)

        # Captura o teclado a cada frame de forma limpa
        tecla = cv2.waitKey(1) & 0xFF

        if circulos is not None:
            circulos = np.uint16(np.around(circulos))
            for i in circulos[0, :]:
                # CORRIGIDO: Desempacotamento correto das coordenadas do array NumPy
                x, y, raio = i[0], i[1], i[2]
                
                # Desenha o indicador visual na tela
                cv2.circle(frame, (x, y), raio, (255, 0, 0), 2)
                cv2.putText(frame, "Pressione 'S' para Salvar", (int(x - 80), int(y - raio - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                if tecla == ord('s') or tecla == ord('S'):
                    textura = extrair_textura_iris(imagem_cinza, x, y, raio)
                    if textura is not None:
                        cadastrar_funcionario(nome, cpf, textura)
                        webcam.release()
                        cv2.destroyAllWindows()
                        sys.exit() # Garante o fechamento limpo do programa
                    else:
                        print("❌ O olho está muito longe. Chegue mais perto da câmera.")

        cv2.imshow("Cadastro de Iris", frame)
        if tecla == ord('q') or tecla == ord('Q'): 
            break

elif opcao == "2":
    print("\nModo Portaria Ativo. Aproxime o olho da câmera...")
    
    contador_frames = 0
    status_texto = "AGUARDANDO PROXIMIDADE"
    cor_status = (255, 255, 255)

    while True:
        sucesso, frame = webcam.read()
        if not sucesso: 
            break
        
        contador_frames += 1
        imagem_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        imagem_borrada = cv2.medianBlur(imagem_cinza, 5)
        circulos = cv2.HoughCircles(imagem_borrada, cv2.HOUGH_GRADIENT, 1, 100, 
                                    param1=50, param2=30, minRadius=25, maxRadius=70)

        if circulos is not None:
            circulos = np.uint16(np.around(circulos))
            for i in circulos[0, :]:
                # CORRIGIDO: Desempacotamento correto das coordenadas do array NumPy
                x, y, raio = i[0], i[1], i[2]
                
                if contador_frames % 5 == 0:
                    textura_atual = extrair_textura_iris(imagem_cinza, x, y, raio)
                    funcionario_encontrado = buscar_iris(textura_atual)
                    
                    if funcionario_encontrado:
                        status_texto = f"DESBLOQUEADO: {funcionario_encontrado}"
                        cor_status = (0, 255, 0) 
                    else:
                        status_texto = "ACESSO NEGADO: DESCONHECIDO"
                        cor_status = (0, 0, 255) 
                
                cv2.circle(frame, (x, y), raio, cor_status, 2)
        else:
            if contador_frames % 15 == 0:
                status_texto = "AGUARDANDO PROXIMIDADE"
                cor_status = (255, 255, 255)

        cv2.rectangle(frame, (0, 0), (640, 40), (0, 0, 0), -1)
        cv2.putText(frame, status_texto, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor_status, 2)

        cv2.imshow("Portaria - Validacao de Acesso", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

webcam.release()
cv2.destroyAllWindows()