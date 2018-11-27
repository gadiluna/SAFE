from pyfiglet import figlet_format


def print_safe():
    a = figlet_format('SAFE', font='starwars')
    print(a)
    print("By Massarelli L., Di Luna G. A., Petroni F., Querzoni L., Baldoni R.")
    print("Please cite: http://arxiv.org/abs/1811.05296 \n")