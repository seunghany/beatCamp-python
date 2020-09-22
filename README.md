Data Dictionary -->> variable = feature = parameter = axis
Variable    Definition  Key
0. survival survival    0 = no, 1 = yes
1. pclass   Ticket class    1 = 1st, 2 = 2nd, 3 = 3rd
2. sex  Sex
3. Age  Age in years
4. sibsp    # of siblings / spuses aborad the Titanic
5. parch    # of parents / children aborad the Titantic
6. ticket   Ticket number
7. fare passenger fare
8. Cabin    Cabin number
9. embarked port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton

Variable Notes
pclass: A poxy for socio-ecnomic status (SES)
1st = Upper
2nd = Middle
3rd = Lower

age = Age is frantional if less than 1. If the age is estimated, it is in the form of xx.5

sibsp: the dataset defines family relations in the way...
sibling: brother, sister, stepbrother, stepsister
spuse = husbamd, wife (mistresses and finaces were ignored)

parch The dataset defines family relations in this way..
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
some children travled sonly with a nanny, therefore parch  = 0 for them
