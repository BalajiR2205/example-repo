pairs = [
    ("sub add { my ($a,$b)=@_; return $a+$b; }",
     "def add(a, b): return a + b"),

    ("sub subtract { my ($a,$b)=@_; return $a-$b; }",
     "def subtract(a, b): return a - b"),

    ("sub multiply { my ($a,$b)=@_; return $a*$b; }",
     "def multiply(a, b): return a * b"),

    ("sub divide { my ($a,$b)=@_; return $a/$b; }",
     "def divide(a, b): return a / b"),

    ("sub pow { my ($a,$b)=@_; return $a**$b; }",
     "def pow(a, b): return a ** b"),
]