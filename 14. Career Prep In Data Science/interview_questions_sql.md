# Interview Questions SQL

### What is the difference between inner and outer join?

- Inner Join: combines rows from two tables where there is a match in both tables. only includes rows that have matching values in both tables. If you have a table of customers and a table of orders, an inner join will return only those customers who have placed orders.

- Outer Join: combines rows from two tables, including all rows from one table, and the matched rows from the other.

  - Left Outer Join: Includes all rows from the left table and matched rows from the right table. If there’s no match, NULLs are returned for columns from the right table.

  - Right Outer Join: Includes all rows from the right table and matched rows from the left table. If there’s no match, NULLs are returned for columns from the left table.

  - Full Outer Join:  Includes all rows from both tables, with NULLs in places where there is no match.

### What