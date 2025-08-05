#Example repository

THis is a sample repo, hello world!

This is a second change following up.

my $transactions = [
    { customer => 'Alice', product => 'Laptop',   qty => 1, price => 1000 },
    { customer => 'Alice', product => 'Mouse',    qty => 2, price => 20   },
    { customer => 'Alice', product => 'Laptop',   qty => 1, price => 1000 },
    { customer => 'Bob',   product => 'Keyboard', qty => 1, price => 60   },
    { customer => 'Alice', product => 'Mouse',    qty => 1, price => 25   },
    { customer => 'Bob',   product => 'Keyboard', qty => 2, price => 55   },
];


%output = (
    'Alice' => {
        'Laptop' => {
            total_qty   => 2,
            total_spent => 2000
        },
        'Mouse' => {
            total_qty   => 3,
            total_spent => 65  # 2*20 + 1*25
        },
    },
    'Bob' => {
        'Keyboard' => {
            total_qty   => 3,
            total_spent => 170  # 1*60 + 2*55
        },
    },
);