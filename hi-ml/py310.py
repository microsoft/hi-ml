class Foo:
    f: int

a = Foo()
a.f = 1
match a:
    case Foo(f=1) as whole:
        print(f"{whole=}")
    case Foo(f=1 as f1):
        print(f"{f1=}")
    case Foo(f=2 as f2):
        print(f"{f2=}")
    case _:
        print("none of the above")

li = [1, 2, 3]
if (n := len(li)) > 1:
    print(f"Too long: {n=}")
print(n)
