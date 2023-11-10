from collections import Counter

class list_color_inertia:
    def __init__(self, track_id, inertia, element):
        self.list = [element]
        self.inertia = inertia
        self.track_id = track_id

    def add_element(self, element):
        if len(self.list) == self.inertia:
            self.list.pop() # Se a lista atingiu a capacidade máxima, remove o último elemento

        self.list.insert(0, element) # Adiciona o novo elemento na primeira posição
    
    def get_tracks_id(self):
        return self.track_id
    
    def get_inertia_color(self):
        if not self.list:
            return [0,0,0]

        counter = Counter(self.list)
        most_common_element = counter.most_common(1)[0][0]
        return most_common_element


inertia = 5

# Exemplo de uso
minha_lista = []
minha_lista.append(list_color_inertia(1, inertia, "a"))
minha_lista[0].add_element("a")
minha_lista[0].add_element("b")
minha_lista[0].add_element("b")
minha_lista.append(list_color_inertia(2, inertia, "b"))
minha_lista[1].add_element("b")
minha_lista[1].add_element("b")
minha_lista[1].add_element("b")
minha_lista[1].add_element("c")
minha_lista.append(list_color_inertia(3, inertia, "c"))
minha_lista[2].add_element("c")
minha_lista[2].add_element("c")
minha_lista[2].add_element("c")
minha_lista[2].add_element("d")

track_id_to_check  = 3
# Encontrar o objeto com o track_id_to_check
found_object = next((obj for obj in minha_lista if obj.get_tracks_id() == track_id_to_check), None)
if found_object is not None:
    found_object.add_element("b")
else:
    minha_lista.append(list_color_inertia(track_id_to_check, inertia, "d"))

found_object = next((obj for obj in minha_lista if obj.get_tracks_id() == track_id_to_check), None)
print(found_object.get_inertia_color())




