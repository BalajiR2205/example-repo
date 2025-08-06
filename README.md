#Example repository

my %tasks = (
    'Frontend' => {
        'High'   => [ 'Fix login bug', 'Improve UI speed' ],
        'Medium' => [ 'Add tooltip to buttons' ],
    },
    'Backend' => {
        'High'   => [ 'Optimize DB queries' ],
        'Low'    => [ 'Refactor config loader' ],
    },
    'DevOps' => {
        'Medium' => [ 'Add monitoring alerts', 'Update Dockerfile' ],
        'High'   => [ 'Migrate to new CI pipeline' ],
    },
);

Output:

[
    {
        team               => 'DevOps',
        total_tasks        => 3,
        high_priority_tasks => 1,
    },
    {
        team               => 'Frontend',
        total_tasks        => 3,
        high_priority_tasks => 2,
    },
    {
        team               => 'Backend',
        total_tasks        => 2,
        high_priority_tasks => 1,
    },
]