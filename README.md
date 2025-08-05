#Example repository


my %departments = (
    'HR' => [ 'Alice', 'Bob' ],
    'IT' => [ 'Charlie', 'David' ],
    'Finance' => [ 'Eve' ],
);

my %salaries = (
    'Alice'   => { base => 50000, bonus => 5000 },
    'Bob'     => { base => 55000, bonus => 6000 },
    'Charlie' => { base => 70000, bonus => 8000 },
    'David'   => { base => 68000, bonus => 7000 },
    'Eve'     => { base => 62000, bonus => 6500 },
);

#output

[
    {
        department => 'HR',
        employees  => [
            { name => 'Alice', total_salary => 55000 },
            { name => 'Bob',   total_salary => 61000 },
        ]
    },
    {
        department => 'IT',
        employees  => [
            { name => 'Charlie', total_salary => 78000 },
            { name => 'David',   total_salary => 75000 },
        ]
    },
    {
        department => 'Finance',
        employees  => [
            { name => 'Eve', total_salary => 68500 },
        ]
    }
]