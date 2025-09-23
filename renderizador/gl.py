#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# pylint: disable=invalid-name

"""
Biblioteca Gráfica / Graphics Library.

Desenvolvido por: <SEU NOME AQUI>
Disciplina: Computação Gráfica
Data: <DATA DE INÍCIO DA IMPLEMENTAÇÃO>
"""

from operator import index
import time         # Para operações com tempo
import gpu          # Simula os recursos de uma GPU
import math         # Funções matemáticas
import numpy as np  # Biblioteca do Numpy

class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800   # largura da tela
    height = 600  # altura da tela
    near = 0.01   # plano de corte próximo
    far = 1000    # plano de corte distante

    @staticmethod
    def setup(width, height, near=0.01, far=1000):
        """Definr parametros para câmera de razão de aspecto, plano próximo e distante."""
        GL.width = width
        GL.height = height
        GL.near = near
        GL.far = far

    @staticmethod
    def polypoint2D(point, colors):
        """Função usada para renderizar Polypoint2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Polypoint2D
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é a
        # coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista e assuma que sempre vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polypoint2D
        # você pode assumir inicialmente o desenho dos pontos com a cor emissiva (emissiveColor)
        emissive_color = colors.get("emissiveColor", [1.0, 1.0, 1.0])
        # Converte cor de (0,1) para (0,255)
        rgb = [int(c * 255) for c in emissive_color]
        for i in range(0, len(point), 2):
            x = int(point[i])
            y = int(point[i + 1])
            gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, rgb)
       
       
    @staticmethod
    def polyline2D(lineSegments, colors):
        """Função usada para renderizar Polyline2D."""
        emissive_color = colors.get("emissiveColor", [1.0, 1.0, 1.0])
        rgb = [int(c * 255) for c in emissive_color]

        def safe_draw_pixel(x, y, color):
            if 0 <= x < GL.width and 0 <= y < GL.height:
                gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, color)

        def draw_line(x0, y0, x1, y1, color):
            # Bresenham com guarda de limites
            x0 = int(x0); y0 = int(y0); x1 = int(x1); y1 = int(y1)

            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            x, y = x0, y0
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1

            # caso degenerado: ponto único
            if dx == 0 and dy == 0:
                safe_draw_pixel(x, y, color)
                return

            if dx >= dy:
                err = dx // 2
                while x != x1:
                    safe_draw_pixel(x, y, color)
                    err -= dy
                    if err < 0:
                        y += sy
                        err += dx
                    x += sx
                safe_draw_pixel(x, y, color)  # último ponto
            else:
                err = dy // 2
                while y != y1:
                    safe_draw_pixel(x, y, color)
                    err -= dx
                    if err < 0:
                        x += sx
                        err += dy
                    y += sy
                safe_draw_pixel(x, y, color)  # último ponto

        for i in range(0, len(lineSegments) - 2, 2):
            x0 = lineSegments[i]
            y0 = lineSegments[i + 1]
            x1 = lineSegments[i + 2]
            y1 = lineSegments[i + 3]
            draw_line(x0, y0, x1, y1, rgb)

        

    @staticmethod
    def circle2D(radius, colors):
        """Função usada para renderizar Circle2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Circle2D
        # Nessa função você receberá um valor de raio e deverá desenhar o contorno de
        # um círculo.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Circle2D
        # você pode assumir o desenho das linhas com a cor emissiva (emissiveColor).

        emissive_color = colors.get("emissiveColor", [1.0, 1.0, 1.0])
        rgb = [int(c * 255) for c in emissive_color]
        cx = GL.width // 2
        cy = GL.height // 2
        num_segments = max(24, int(2 * math.pi * radius))  # ajusta para suavidade

        def draw_line(x0, y0, x1, y1, color):
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            x, y = x0, y0
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1

            if dx > dy:
                err = dx // 2
                while x != x1:
                    if 0 <= x < GL.width and 0 <= y < GL.height:
                        gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, color)
                    err -= dy
                    if err < 0:
                        y += sy
                        err += dx
                    x += sx
                if 0 <= x < GL.width and 0 <= y < GL.height:
                    gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, color)
            else:
                err = dy // 2
                while y != y1:
                    if 0 <= x < GL.width and 0 <= y < GL.height:
                        gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, color)
                    err -= dx
                    if err < 0:
                        x += sx
                        err += dy
                    y += sy
                if 0 <= x < GL.width and 0 <= y < GL.height:
                    gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, color)

        # Desenha o círculo ponto a ponto
        for i in range(num_segments):
            theta1 = 2 * math.pi * i / num_segments
            theta2 = 2 * math.pi * (i + 1) / num_segments
            x0 = int(cx + radius * math.cos(theta1))
            y0 = int(cy + radius * math.sin(theta1))
            x1 = int(cx + radius * math.cos(theta2))
            y1 = int(cy + radius * math.sin(theta2))
            draw_line(x0, y0, x1, y1, rgb)

    @staticmethod
    def triangleSet2D(vertices, colors):
        """Função usada para renderizar TriangleSet2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#TriangleSet2D
        # Nessa função você receberá os vertices de um triângulo no parâmetro vertices,
        # esses pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o
        # valor da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto.
        # Já point[2] é a coordenada x do segundo ponto e assim por diante. Assuma que a
        # quantidade de pontos é sempre multiplo de 3, ou seja, 6 valores ou 12 valores, etc.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o TriangleSet2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).
        fb_dim = (400, 600)  # ajuste conforme seu framebuffer
        emissive_color = colors.get("emissiveColor", [1.0, 1.0, 1.0])
        rgb = [int(c * 255) for c in emissive_color]

        h, w = fb_dim  # altura, largura

        def draw_pixel_safe(x, y, color):
            if 0 <= x < w and 0 <= y < h:
                gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, color)

        def draw_line(x0, y0, x1, y1, color):
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            x, y = x0, y0
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            if dx > dy:
                err = dx // 2
                while x != x1:
                    draw_pixel_safe(x, y, color)
                    err -= dy
                    if err < 0:
                        y += sy
                        err += dx
                    x += sx
                draw_pixel_safe(x, y, color)
            else:
                err = dy // 2
                while y != y1:
                    draw_pixel_safe(x, y, color)
                    err -= dx
                    if err < 0:
                        x += sx
                        err += dy
                    y += sy
                draw_pixel_safe(x, y, color)

        def fill_triangle(v0, v1, v2, color):
            # Ordena vértices por y
            vertices_sorted = sorted([v0, v1, v2], key=lambda v: v[1])
            x0, y0 = vertices_sorted[0]
            x1, y1 = vertices_sorted[1]
            x2, y2 = vertices_sorted[2]

            def edge_interp(y, y0, x0, y1, x1):
                if y1 == y0:
                    return x0
                return int(x0 + (x1 - x0) * (y - y0) / (y1 - y0))

            for y in range(y0, y2 + 1):
                if y < y1:
                    xa = edge_interp(y, y0, x0, y1, x1)
                    xb = edge_interp(y, y0, x0, y2, x2)
                else:
                    xa = edge_interp(y, y1, x1, y2, x2)
                    xb = edge_interp(y, y0, x0, y2, x2)

                if xa > xb:
                    xa, xb = xb, xa
                for x in range(xa, xb + 1):
                    draw_pixel_safe(x, y, color)

        for i in range(0, len(vertices), 6):
            v0 = (int(vertices[i]), int(vertices[i+1]))
            v1 = (int(vertices[i+2]), int(vertices[i+3]))
            v2 = (int(vertices[i+4]), int(vertices[i+5]))

            # Preenche triângulo
            fill_triangle(v0, v1, v2, rgb)

            # Desenha bordas
            draw_line(*v0, *v1, rgb)
            draw_line(*v1, *v2, rgb)
            draw_line(*v2, *v0, rgb)




    @staticmethod
    def triangleSet(point, colors, vertex_colors=None, vertex_uvs=None, texture=None):
        """Desenha triângulos 3D com interpolação perspectiva de cor e textura."""
    # print removido
        w, h = GL.width, GL.height
        if not hasattr(GL, 'z_buffer'):
            GL.z_buffer = np.full((h, w), GL.far, dtype=float)

        def project_vertex(v):
            vec = np.array([v[0], v[1], v[2], 1.0])
            # print removido
            if hasattr(GL, 'transform_stack') and GL.transform_stack:
                vec = GL.transform_stack[-1] @ vec
                # print removido
            vec = GL.view_matrix @ vec
            # print removido
            vec = GL.projection_matrix @ vec
            # print removido
            if vec[3] != 0:
                vec /= vec[3]
            sx = int((vec[0]+1)*w/2)
            sy = int((1-vec[1])*h/2)
            sz = (vec[2]+1)/2
            # print removido
            return [sx, sy, sz, 1.0/vec[3] if vec[3] != 0 else 1.0]

        def barycentric_coords(px, py, v0, v1, v2):
            x0, y0 = v0[0], v0[1]
            x1, y1 = v1[0], v1[1]
            x2, y2 = v2[0], v2[1]
            denom = (y1 - y2)*(x0 - x2) + (x2 - x1)*(y0 - y2)
            if denom == 0:
                return 0,0,0
            w0 = ((y1 - y2)*(px - x2) + (x2 - x1)*(py - y2)) / denom
            w1 = ((y2 - y0)*(px - x2) + (x0 - x2)*(py - y2)) / denom
            w2 = 1 - w0 - w1
            return w0, w1, w2

        def safe_draw_pixel(x, y, z, color):
            if 0 <= x < w and 0 <= y < h:
                if z < GL.z_buffer[y, x]:
                    gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, color)
                    GL.z_buffer[y, x] = z

        def sample_texture(u, v, texture, flip_v=True, flip_u=False):
            if texture is None:
                return None
            import numpy as np
            from renderizador import gpu
            # Se for lista (ex: ['insper.png']), carrega a textura
            if isinstance(texture, list):
                if len(texture) > 0 and isinstance(texture[0], str):
                    texture = gpu.GPU.load_texture(texture[0].strip('" '))
                else:
                    return None
            if not hasattr(texture, 'shape') or len(texture.shape) < 2:
                return None
            h, w = texture.shape[0], texture.shape[1]
            # Clamp UV
            u = max(0, min(1, u))
            v = max(0, min(1, v))
            if flip_u:
                u = 1 - u
            if flip_v:
                v = 1 - v
            x = int(u * (w-1))
            y = int(v * (h-1))
            color = texture[y, x]
            if hasattr(color, '__len__') and len(color) == 4:
                # Retorna RGBA
                return color.tolist() if hasattr(color, 'tolist') else list(color)
            return color.tolist() if hasattr(color, 'tolist') else list(color)

        def fill_triangle(v0, v1, v2, c0, c1, c2, uv0, uv1, uv2):
            minx, maxx = max(0, min(v0[0], v1[0], v2[0])), min(w-1, max(v0[0], v1[0], v2[0]))
            miny, maxy = max(0, min(v0[1], v1[1], v2[1])), min(h-1, max(v0[1], v1[1], v2[1]))
            light = GL.get_directional_light() if hasattr(GL, 'get_directional_light') else None
            # Parâmetros de iluminação
            if light:
                L = np.array(light['direction'])
                L = L / np.linalg.norm(L)
                light_color = np.array(light['color'])*255.0 if max(light['color']) <= 1.0 else np.array(light['color'])
                intensity = light['intensity']
                ambient_intensity = light['ambientIntensity']
            else:
                L = np.array([0,0,-1])
                light_color = np.array([255,255,255])
                intensity = 5.0
                ambient_intensity = 0.8
            for y in range(miny, maxy+1):
                for x in range(minx, maxx+1):
                    w0, w1, w2 = barycentric_coords(x, y, v0, v1, v2)
                    if w0 >= 0 and w1 >= 0 and w2 >= 0:
                        z0, z1, z2 = v0[2], v1[2], v2[2]
                        inv_w0, inv_w1, inv_w2 = v0[3], v1[3], v2[3]
                        inv_w = w0*inv_w0 + w1*inv_w1 + w2*inv_w2
                        if inv_w == 0:
                            continue
                        w0p = (w0*inv_w0) / inv_w
                        w1p = (w1*inv_w1) / inv_w
                        w2p = (w2*inv_w2) / inv_w
                        # Interpola cor base
                        if c0 is not None and c1 is not None and c2 is not None:
                            base_color = np.array([w0p*c0[i] + w1p*c1[i] + w2p*c2[i] for i in range(3)])
                        else:
                            # Usa diffuseColor corretamente
                            base_color = np.array([c*255 for c in colors.get("diffuseColor", [1,1,1])])
                        # Interpola UV
                        if uv0 is not None and uv1 is not None and uv2 is not None and texture is not None:
                            u = w0p*uv0[0] + w1p*uv1[0] + w2p*uv2[0]
                            v_ = w0p*uv0[1] + w1p*uv1[1] + w2p*uv2[1]
                            tex_color = sample_texture(u, v_, texture, flip_v=True, flip_u=False)
                            if tex_color is not None:
                                if len(tex_color) == 4:
                                    alpha = tex_color[3] / 255.0
                                    if alpha < 1.0:
                                        from renderizador import gpu
                                        try:
                                            bg = gpu.GPU.read_pixel([x, y], gpu.GPU.RGB8)
                                        except Exception:
                                            bg = [0, 0, 0]
                                        base_color = np.array([
                                            tex_color[i]*alpha + bg[i]*(1-alpha) for i in range(3)
                                        ])
                                    else:
                                        base_color = np.array(tex_color[:3])
                                else:
                                    base_color = np.array(tex_color[:3])
                        # Normal interpolada (aqui assume normal da face para simplicidade)
                        v01 = np.array(v1[:3]) - np.array(v0[:3])
                        v02 = np.array(v2[:3]) - np.array(v0[:3])
                        N = np.cross(v01, v02)
                        if np.linalg.norm(N) > 0:
                            N = N / np.linalg.norm(N)
                        else:
                            N = np.array([0,0,1])
                        # Garante que a normal aponte para a câmera (z negativo)
                        if N[2] > 0:
                            N = -N
                        # Vetor para câmera (assume câmera em [0,0,0])
                        P = w0p*np.array(v0[:3]) + w1p*np.array(v1[:3]) + w2p*np.array(v2[:3])
                        V = -P / (np.linalg.norm(P)+1e-6)
                        # Iluminação ambiente
                        ambient = ambient_intensity * base_color
                        # Iluminação difusa
                        diff = max(0, np.dot(N, -L))
                        diffuse = intensity * diff * light_color * base_color / 255.0
                        # Iluminação especular
                        specular = np.zeros(3)
                        shininess = 32
                        if diff > 0:
                            R = 2 * np.dot(N, -L) * N + L
                            R = R / (np.linalg.norm(R)+1e-6)
                            spec = max(0, np.dot(R, V)) ** shininess
                            specular = intensity * spec * light_color * 255.0
                        color = ambient + diffuse + specular
                        color = np.clip(color, 0, 255).astype(int)
                        # Transparência RGBA
                        if 'transparency' in colors and colors['transparency'] > 0.0:
                            alpha = 1.0 - colors['transparency']
                            if alpha < 1.0:
                                from renderizador import gpu
                                try:
                                    bg = gpu.GPU.read_pixel([x, y], gpu.GPU.RGB8)
                                except Exception:
                                    bg = [0, 0, 0]
                                color = [
                                    int(color[i]*alpha + bg[i]*(1-alpha)) for i in range(3)
                                ]
                        z = w0p*z0 + w1p*z1 + w2p*z2
                        safe_draw_pixel(x, y, z, color)

        n = len(point) // 3
        for i in range(0, len(point), 9):
            v0 = project_vertex(point[i:i+3])
            v1 = project_vertex(point[i+3:i+6])
            v2 = project_vertex(point[i+6:i+9])
            if vertex_colors:
                c0, c1, c2 = vertex_colors
            else:
                c0 = c1 = c2 = None
            if vertex_uvs:
                uv0, uv1, uv2 = vertex_uvs
            else:
                uv0 = uv1 = uv2 = None
            fill_triangle(v0, v1, v2, c0, c1, c2, uv0, uv1, uv2)


    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Configura a câmera e a matriz de projeção."""
        
        GL.camera_position = np.array(position, dtype=float)
        GL.camera_orientation = np.array(orientation, dtype=float)
        GL.fov = fieldOfView

        # Projeção perspectiva
        aspect = GL.width / GL.height
        near = GL.near
        far = GL.far
        f = 1 / math.tan(fieldOfView / 2)

        GL.projection_matrix = np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near)/(near - far), (2*far*near)/(near - far)],
            [0, 0, -1, 0]
        ], dtype=float)

        # Orientação da câmera: eixo e ângulo
        ax, ay, az, angle = orientation
        c = math.cos(angle)
        s = math.sin(angle)
        t = 1 - c

        # Matriz de rotação (Rodrigues)
        R = np.array([
            [t*ax*ax + c,     t*ax*ay - s*az, t*ax*az + s*ay],
            [t*ax*ay + s*az, t*ay*ay + c,     t*ay*az - s*ax],
            [t*ax*az - s*ay, t*ay*az + s*ax, t*az*az + c]
        ], dtype=float)

        # Matriz de translação
        T = np.identity(4)
        T[:3, 3] = -GL.camera_position

        # Matriz de visualização completa
        GL.view_matrix = np.identity(4)
        GL.view_matrix[:3, :3] = R.T  # Transposta da rotação
        GL.view_matrix = GL.view_matrix @ T



    @staticmethod
    def transform_in(translation=None, scale=None, rotation=None):
        """
        Aplica uma transformação de modelo 3D (Transform node do X3D) usando matriz 4x4.

        translation: [x, y, z] deslocamento do objeto
        scale: [sx, sy, sz] escala do objeto
        rotation: [ax, ay, az, angle] rotação ao redor do eixo (ax, ay, az) por 'angle' radianos
        """
        S = np.eye(4, dtype=float)

        if scale:
            S[0, 0] = float(scale[0])
            S[1, 1] = float(scale[1])
            S[2, 2] = float(scale[2])

        R4 = np.eye(4, dtype=float)
        if rotation:
            ax, ay, az, angle = rotation
            axis = np.array([ax, ay, az], dtype=float)
            norm = np.linalg.norm(axis)
            if norm == 0:
                # rotação nula (identidade)
                R3 = np.eye(3, dtype=float)
            else:
                ux, uy, uz = axis / norm
                c = math.cos(angle)
                s = math.sin(angle)
                t = 1 - c
                R3 = np.array([
                    [t*ux*ux + c,     t*ux*uy - s*uz, t*ux*uz + s*uy],
                    [t*ux*uy + s*uz, t*uy*uy + c,     t*uy*uz - s*ux],
                    [t*ux*uz - s*uy, t*uy*uz + s*ux, t*uz*uz + c]
                ], dtype=float)
            R4[0:3, 0:3] = R3

        T = np.eye(4, dtype=float)
        if translation:
            T[0, 3] = float(translation[0])
            T[1, 3] = float(translation[1])
            T[2, 3] = float(translation[2])

        # matriz local: primeiro escala, depois rotaciona, depois traduz
        M_local = T @ R4 @ S

        # inicializa pilha se necessário
        if not hasattr(GL, "transform_stack"):
            GL.transform_stack = []

        # Se já existe uma transformação pai, concatena: parent @ M_local
        if GL.transform_stack:
            M_world = GL.transform_stack[-1] @ M_local
        else:
            M_world = M_local

        # empurra a matriz composta (world) na pilha e atualiza model_matrix
        GL.transform_stack.append(M_world)
        GL.model_matrix = M_world



    @staticmethod
    def transform_out():
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_out será chamada quando se sair em um nó X3D do tipo Transform do
        # grafo de cena. Não são passados valores, porém quando se sai de um nó transform se
        # deverá recuperar a matriz de transformação dos modelos do mundo da estrutura de
        # pilha implementada.
        if hasattr(GL, "transform_stack") and GL.transform_stack:
            GL.transform_stack.pop()
            if GL.transform_stack:
                GL.model_matrix = GL.transform_stack[-1]
            else:
                GL.model_matrix = np.eye(4, dtype=float)
        else:
            # pilha já vazia — mantém identidade e emite aviso
            GL.model_matrix = np.eye(4, dtype=float)

    @staticmethod
    def triangleStripSet(point, stripCount, colors):
        """Função usada para renderizar TriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleStripSet
        # A função triangleStripSet é usada para desenhar tiras de triângulos interconectados,
        # você receberá as coordenadas dos pontos no parâmetro point, esses pontos são uma
        # lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x
        # do primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e assim
        # por diante. No TriangleStripSet a quantidade de vértices a serem usados é informado
        # em uma lista chamada stripCount (perceba que é uma lista). Ligue os vértices na ordem,
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.
        offset = 0
        for strip in stripCount:
            # cada tira com N vértices gera (N-2) triângulos
            for i in range(strip - 2):
                # pega os 3 vértices consecutivos
                v1 = point[(offset + i) * 3:(offset + i) * 3 + 3]
                v2 = point[(offset + i + 1) * 3:(offset + i + 1) * 3 + 3]
                v3 = point[(offset + i + 2) * 3:(offset + i + 2) * 3 + 3]

                # alterna a ordem para manter a orientação consistente
                if i % 2 == 0:
                    tri = v1 + v2 + v3
                else:
                    tri = v2 + v1 + v3

                # desenha usando o rasterizador 3D já existente
                GL.triangleSet(tri, colors)

            offset += strip


        
    @staticmethod
    def indexedTriangleStripSet(point, index, colors):
        """Função usada para renderizar IndexedTriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#IndexedTriangleStripSet
        # A função indexedTriangleStripSet é usada para desenhar tiras de triângulos
        # interconectados, você receberá as coordenadas dos pontos no parâmetro point, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor
        # da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto, point[2]
        # o valor z da coordenada z do primeiro ponto. Já point[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedTriangleStripSet uma lista informando
        # como conectar os vértices é informada em index, o valor -1 indica que a lista
        # acabou. A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        strip = []
        for idx in index:
            if idx == -1:
                # processa a strip acumulada
                for i in range(len(strip) - 2):
                    v1 = point[strip[i] * 3:strip[i] * 3 + 3]
                    v2 = point[strip[i + 1] * 3:strip[i + 1] * 3 + 3]
                    v3 = point[strip[i + 2] * 3:strip[i + 2] * 3 + 3]

                    if i % 2 == 0:
                        tri = v1 + v2 + v3
                    else:
                        tri = v2 + v1 + v3

                    GL.triangleSet(tri, colors)
                strip = []
            else:
                strip.append(idx)

    @staticmethod
    def indexedFaceSet(coord, coordIndex, colorPerVertex, color, colorIndex,
                    texCoord, texCoordIndex, colors, current_texture):
        """Renderiza um IndexedFaceSet usando triangleSet da GPU, com suporte a interpolação de cor por vértice."""

        def normalize_color(col):
            if col is None:
                return [1.0, 1.0, 1.0]
            col = list(col)
            if len(col) >= 3:
                col = col[:3]
            else:
                while len(col) < 3:
                    col.append(1.0)
            mx = max(col)
            if mx <= 1.0:
                return [max(0.0, min(1.0, float(c))) for c in col]
            return [max(0.0, min(1.0, float(c) / 255.0)) for c in col]

        def get_color_for_index(idx):
            if colorPerVertex:
                if isinstance(color, (list, tuple)) and len(color) >= (idx + 1) * 3:
                    return normalize_color(color[idx * 3: idx * 3 + 3])
                if isinstance(colorIndex, (list, tuple)) and isinstance(color, (list, tuple)):
                    if idx < len(colorIndex):
                        ci = colorIndex[idx]
                        if ci >= 0 and len(color) >= (ci + 1) * 3:
                            return normalize_color(color[ci * 3: ci * 3 + 3])
            if isinstance(color, (list, tuple)) and len(color) >= 3:
                return normalize_color(color)
            if isinstance(colors, dict):
                return normalize_color(colors.get("emissiveColor", [1.0, 1.0, 1.0]))
            return [1.0, 1.0, 1.0]

        def get_uv_for_triangle(poly_start, tri_indices):
            # poly_start: início do polígono atual em texCoordIndex
            # tri_indices: lista de 3 índices (posição dos vértices do triângulo dentro do polígono)
            if texCoord is None:
                return [None, None, None]
            if texCoordIndex and isinstance(texCoordIndex, (list, tuple)):
                uvs = []
                for pos in tri_indices:
                    uv_idx = texCoordIndex[poly_start + pos] if (poly_start + pos) < len(texCoordIndex) else None
                    if uv_idx is not None and uv_idx >= 0 and len(texCoord) >= (uv_idx + 1) * 2:
                        uvs.append([texCoord[uv_idx * 2], texCoord[uv_idx * 2 + 1]])
                    else:
                        uvs.append(None)
                return uvs
            # Caso contrário, use o mesmo índice de coordIndex
            uvs = []
            for idx in tri_indices:
                v_idx = idx
                if v_idx is not None and len(texCoord) >= (v_idx + 1) * 2:
                    uvs.append([texCoord[v_idx * 2], texCoord[v_idx * 2 + 1]])
                else:
                    uvs.append(None)
            return uvs

        face = []
        poly_start_tc = 0  # início do polígono atual em texCoordIndex
        for idx in coordIndex:
            if idx == -1:
                if len(face) >= 3:
                    for i in range(1, len(face) - 1):
                        v1 = coord[face[0]*3:face[0]*3+3]
                        v2 = coord[face[i]*3:face[i]*3+3]
                        v3 = coord[face[i+1]*3:face[i+1]*3+3]
                        tri = v1 + v2 + v3

                        # Cores
                        if colorPerVertex:
                            c0 = get_color_for_index(face[0])
                            c1 = get_color_for_index(face[i])
                            c2 = get_color_for_index(face[i+1])
                            vertex_colors = [
                                [int(255*c0[0]), int(255*c0[1]), int(255*c0[2])],
                                [int(255*c1[0]), int(255*c1[1]), int(255*c1[2])],
                                [int(255*c2[0]), int(255*c2[1]), int(255*c2[2])],
                            ]
                        else:
                            c = get_color_for_index(0)
                            vertex_colors = None
                        # UVs: calcula os índices corretos para cada triângulo
                        tri_indices = [0, i, i+1]
                        uv0, uv1, uv2 = get_uv_for_triangle(poly_start_tc, tri_indices)
                        vertex_uvs = [uv0, uv1, uv2] if uv0 and uv1 and uv2 else None
                        # Chama triangleSet com textura
                        GL.triangleSet(tri, colors, vertex_colors=vertex_colors, vertex_uvs=vertex_uvs, texture=current_texture)

                # Atualiza poly_start_tc para o próximo polígono
                if texCoordIndex and isinstance(texCoordIndex, (list, tuple)):
                    poly_start_tc += len(face) + 1
                face = []
            else:
                face.append(idx)


    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Box
        # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
        # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
        # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
        # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
        # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Box : size = {0}".format(size)) # imprime no terminal pontos
        print("Box : colors = {0}".format(colors)) # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Sphere
        # A função sphere é usada para desenhar esferas na cena. O esfera é centrada no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da esfera que está sendo criada. Para desenha essa esfera você vai
        # precisar tesselar ela em triângulos, para isso encontre os vértices e defina
        # os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Sphere : radius = {0}".format(radius)) # imprime no terminal o raio da esfera
        print("Sphere : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def cone(bottomRadius, height, colors):
        """Função usada para renderizar Cones."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cone
        # A função cone é usada para desenhar cones na cena. O cone é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento bottomRadius especifica o
        # raio da base do cone e o argumento height especifica a altura do cone.
        # O cone é alinhado com o eixo Y local. O cone é fechado por padrão na base.
        # Para desenha esse cone você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Cone : bottomRadius = {0}".format(bottomRadius)) # imprime no terminal o raio da base do cone
        print("Cone : height = {0}".format(height)) # imprime no terminal a altura do cone
        print("Cone : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def cylinder(radius, height, colors):
        """Função usada para renderizar Cilindros."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cylinder
        # A função cylinder é usada para desenhar cilindros na cena. O cilindro é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da base do cilindro e o argumento height especifica a altura do cilindro.
        # O cilindro é alinhado com o eixo Y local. O cilindro é fechado por padrão em ambas as extremidades.
        # Para desenha esse cilindro você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Cylinder : radius = {0}".format(radius)) # imprime no terminal o raio do cilindro
        print("Cylinder : height = {0}".format(height)) # imprime no terminal a altura do cilindro
        print("Cylinder : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def navigationInfo(headlight):
        """Configura a headlight (luz da câmera) caso esteja ativada."""
        
        if headlight:
            # Define a luz direcional presa à câmera
            light = {
                "type": "directional",
                "ambientIntensity": 0.0,
                "intensity": 1.0,
                "color": [1.0, 1.0, 1.0],
                "direction": [0.0, 0.0, -1.0]  # olhando para frente
            }
            return light
        else:
            # Sem headlight
            return None


    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction, normal=None):
        """Calcula a contribuição de uma luz direcional em um ponto com normal fornecida."""

        # Normalizar direção e normal
        L = np.array(direction, dtype=float)
        L = L / np.linalg.norm(L)  # direção da luz
        if normal is not None:
            N = np.array(normal, dtype=float)
            N = N / np.linalg.norm(N)  # normal da superfície
        else:
            N = np.array([0,0,1], dtype=float)

        # Luz ambiente (afeta toda superfície igualmente)
        ambient = np.array(color) * ambientIntensity

        # Luz difusa (modelo de Lambert)
        diff = max(np.dot(N, -L), 0.0)  # -L porque a luz vem na direção oposta
        diffuse = np.array(color) * intensity * diff

        # Resultado final (clamp entre 0 e 1)
        result = ambient + diffuse
        result = np.clip(result, 0, 1)

        return result.tolist()

    @staticmethod
    def get_directional_light():
        return getattr(GL, '_directional_light', None)

    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        """Luz pontual."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#PointLight
        # Fonte de luz pontual em um local 3D no sistema de coordenadas local. Uma fonte
        # de luz pontual emite luz igualmente em todas as direções; ou seja, é omnidirecional.
        # Possui os campos básicos ambientIntensity, cor, intensidade. Um nó PointLight ilumina
        # a geometria em um raio de sua localização. O campo do raio deve ser maior ou igual a
        # zero. A iluminação do nó PointLight diminui com a distância especificada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("PointLight : ambientIntensity = {0}".format(ambientIntensity))
        print("PointLight : color = {0}".format(color)) # imprime no terminal
        print("PointLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("PointLight : location = {0}".format(location)) # imprime no terminal

    @staticmethod
    def fog(visibilityRange, color):
        """Névoa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/environmentalEffects.html#Fog
        # O nó Fog fornece uma maneira de simular efeitos atmosféricos combinando objetos
        # com a cor especificada pelo campo de cores com base nas distâncias dos
        # vários objetos ao visualizador. A visibilidadeRange especifica a distância no
        # sistema de coordenadas local na qual os objetos são totalmente obscurecidos
        # pela névoa. Os objetos localizados fora de visibilityRange do visualizador são
        # desenhados com uma cor de cor constante. Objetos muito próximos do visualizador
        # são muito pouco misturados com a cor do nevoeiro.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Fog : color = {0}".format(color)) # imprime no terminal
        print("Fog : visibilityRange = {0}".format(visibilityRange))

    @staticmethod
    def timeSensor(cycleInterval, loop):
        """Gera eventos conforme o tempo passa (fração normalizada 0..1)."""

        if cycleInterval <= 0:
            raise ValueError("cycleInterval deve ser maior que zero")

        # tempo atual em segundos
        epoch = time.time()

        if loop:
            # fração cíclica (sempre entre 0 e 1)
            fraction_changed = (epoch % cycleInterval) / cycleInterval
        else:
            # tempo desde o início (epoch inicial fixo pode ser usado se precisar de consistência)
            elapsed = epoch % (cycleInterval * 1000000)  # só para não crescer infinito
            if elapsed >= cycleInterval:
                fraction_changed = 1.0
            else:
                fraction_changed = elapsed / cycleInterval

        return fraction_changed

    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """Interpola não linearmente entre uma lista de vetores 3D (Catmull-Rom)."""
        # Validações básicas
        try:
            set_fraction = float(set_fraction)
        except Exception:
            return [0.0, 0.0, 0.0]

        if not key or len(key) < 2 or not keyValue or len(keyValue) < 2:
            return [0.0, 0.0, 0.0]

        # Garantir limites
        if set_fraction <= key[0]:
            return list(keyValue[0])
        if set_fraction >= key[-1]:
            return list(keyValue[-1])

        # Decide se vamos tratar como loop (closed) — só se os pontos inicial e final forem idênticos
        loop = bool(closed) and (len(keyValue) >= 2) and (tuple(keyValue[0]) == tuple(keyValue[-1]))

        # encontra intervalo i tal que key[i] <= set_fraction <= key[i+1]
        i = 0
        while i < len(key) - 1 and not (key[i] <= set_fraction <= key[i + 1]):
            i += 1
        # segurança
        if i >= len(key) - 1:
            return list(keyValue[-1])

        # parâmetro local t em [0,1]
        denom = (key[i + 1] - key[i])
        t = 0.0 if denom == 0 else (set_fraction - key[i]) / denom

        # pontos de controle para Catmull-Rom: p0, p1, p2, p3
        def get_point(idx):
            n = len(keyValue)
            if loop:
                return keyValue[idx % n]
            else:
                # clamp
                idx_clamped = max(0, min(idx, n - 1))
                return keyValue[idx_clamped]

        p1 = get_point(i)
        p2 = get_point(i + 1)
        p0 = get_point(i - 1)
        p3 = get_point(i + 2)

        # Catmull-Rom spline (component-wise)
        def catmull_rom_comp(p0c, p1c, p2c, p3c, t):
            t2 = t * t
            t3 = t2 * t
            return 0.5 * (
                (2 * p1c) +
                (-p0c + p2c) * t +
                (2*p0c - 5*p1c + 4*p2c - p3c) * t2 +
                (-p0c + 3*p1c - 3*p2c + p3c) * t3
            )

        x = catmull_rom_comp(p0[0], p1[0], p2[0], p3[0], t)
        y = catmull_rom_comp(p0[1], p1[1], p2[1], p3[1], t)
        z = catmull_rom_comp(p0[2], p1[2], p2[2], p3[2], t)

        return [float(x), float(y), float(z)]

    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """Interpola rotações absolutas usando SLERP (axis-angle -> quaternion -> slerp -> axis-angle)."""
        # Retorna [ax, ay, az, angle]
        try:
            set_fraction = float(set_fraction)
        except Exception:
            return [0.0, 0.0, 1.0, 0.0]

        if not key or len(key) < 2 or not keyValue or len(keyValue) < 2:
            return [0.0, 0.0, 1.0, 0.0]

        # limites
        if set_fraction <= key[0]:
            return list(keyValue[0])
        if set_fraction >= key[-1]:
            return list(keyValue[-1])

        # encontra intervalo
        i = 0
        while i < len(key) - 1 and not (key[i] <= set_fraction <= key[i + 1]):
            i += 1
        if i >= len(key) - 1:
            return list(keyValue[-1])

        denom = (key[i + 1] - key[i])
        t = 0.0 if denom == 0 else (set_fraction - key[i]) / denom

        r1 = keyValue[i]
        r2 = keyValue[i + 1]

        axis1 = [float(r1[0]), float(r1[1]), float(r1[2])]
        angle1 = float(r1[3])
        axis2 = [float(r2[0]), float(r2[1]), float(r2[2])]
        angle2 = float(r2[3])

        # converte axis-angle -> quaternion [x,y,z,w]
        def axis_angle_to_quat(axis, angle):
            ax, ay, az = axis
            norm = math.sqrt(ax*ax + ay*ay + az*az)
            if norm == 0.0:
                return [0.0, 0.0, 0.0, 1.0]
            ax, ay, az = ax / norm, ay / norm, az / norm
            s = math.sin(angle / 2.0)
            w = math.cos(angle / 2.0)
            return [ax * s, ay * s, az * s, w]

        # quaternion -> axis-angle
        def quat_to_axis_angle(q):
            x, y, z, w = q
            # normalizar quaternion
            mag = math.sqrt(x*x + y*y + z*z + w*w)
            if mag == 0.0:
                return [0.0, 0.0, 1.0, 0.0]
            x /= mag; y /= mag; z /= mag; w /= mag
            angle = 2.0 * math.acos(max(-1.0, min(1.0, w)))
            s = math.sqrt(max(0.0, 1.0 - w*w))
            if s < 1e-6:
                # ângulo muito pequeno -> eixo arbitrário
                return [1.0, 0.0, 0.0, 0.0]
            return [x / s, y / s, z / s, angle]

        # SLERP entre q1 e q2
        def slerp(q1, q2, t):
            dot = q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3]
            # se dot < 0, inverte q2 para tomar o menor arco
            if dot < 0.0:
                q2 = [-q2[0], -q2[1], -q2[2], -q2[3]]
                dot = -dot
            # se quase colinear, faz interpolação linear e normaliza
            if dot > 0.9995:
                res = [q1[i] + t*(q2[i] - q1[i]) for i in range(4)]
                mag = math.sqrt(sum(c*c for c in res))
                return [c/mag for c in res]
            theta_0 = math.acos(max(-1.0, min(1.0, dot)))
            theta = theta_0 * t
            sin_theta = math.sin(theta)
            sin_theta_0 = math.sin(theta_0)
            s1 = math.cos(theta) - dot * sin_theta / sin_theta_0
            s2 = sin_theta / sin_theta_0
            return [s1 * q1[0] + s2 * q2[0],
                    s1 * q1[1] + s2 * q2[1],
                    s1 * q1[2] + s2 * q2[2],
                    s1 * q1[3] + s2 * q2[3]]

        q1 = axis_angle_to_quat(axis1, angle1)
        q2 = axis_angle_to_quat(axis2, angle2)
        q = slerp(q1, q2, t)
        out = quat_to_axis_angle(q)
        return [float(out[0]), float(out[1]), float(out[2]), float(out[3])]

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""
