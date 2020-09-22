class Contact:
    def __init__(self, name, phone, email, addr):
        pass
    
    def print_info(self):
        print("Name: {self.name}, Phone# {self.phone}, Email: {self.email}, Address: {self.addr}")

    @staticmethod
    def set_contact():
        name = input('Name') # 나머지 완성
        contact = Contact(name)
        return contact
    @staticmethod
    def get_contact(clist):
        for i in clist:
            i.print_info()

    @staticmethod
    def del_contact(clist, name):
        for i, t in enumerate(clist):
            if t.name == name:
                del clist[i]

    @staticmethod
    def print_menu():
        print('1 Input Contact')
        print('2 Print Contact')
        print('3 Delete Contact')
        print('4 Exit')
        menu = input('Choose Option:')
        return menu
    @staticmethod
    def run():
        clist =[]    
        while 1:
            menu = Contact.print_menu()
            if menu =='1':
                t = Contact.print_contact()
                clist.append(t)
            if menu =='2':
                Contact.get_contact(clist) # Static method 는 클래스를 직접 접근함
            if menu == '3':
                name = input('input contact you want to delete')
            elif menu =='4':
                print('Exit')
                break
if __name__ == '__main__':
    Contact.run()