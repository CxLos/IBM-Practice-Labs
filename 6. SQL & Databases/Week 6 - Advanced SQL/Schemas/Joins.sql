DROP DATABASE IF EXISTS ibm8_db;
CREATE DATABASE ibm8_db;

USE ibm8_db;

DROP TABLE IF EXISTS EMPLOYEES;
DROP TABLE IF EXISTS JOB_HISTORY;
DROP TABLE IF EXISTS JOBS;
DROP TABLE IF EXISTS DEPARTMENTS;
DROP TABLE IF EXISTS LOCATIONS;



CREATE TABLE EMPLOYEES (
                          EMP_ID CHAR(9) NOT NULL,
                          F_NAME VARCHAR(15) NOT NULL,
                          L_NAME VARCHAR(15) NOT NULL,
                          SSN CHAR(9),
                          B_DATE DATE,
                          SEX CHAR,
                          ADDRESS VARCHAR(30),
                          JOB_ID CHAR(9),
                          SALARY DECIMAL(10,2),
                          MANAGER_ID CHAR(9),
                          DEP_ID CHAR(9) NOT NULL,
                          PRIMARY KEY (EMP_ID)
                        );

CREATE TABLE JOB_HISTORY (
                            EMPL_ID CHAR(9) NOT NULL,
                            START_DATE DATE,
                            JOBS_ID CHAR(9) NOT NULL,
                            DEPT_ID CHAR(9),
                            PRIMARY KEY (EMPL_ID,JOBS_ID)
                          );

CREATE TABLE JOBS (
                    JOB_IDENT CHAR(9) NOT NULL,
                    JOB_TITLE VARCHAR(30) ,
                    MIN_SALARY DECIMAL(10,2),
                    MAX_SALARY DECIMAL(10,2),
                    PRIMARY KEY (JOB_IDENT)
                  );

CREATE TABLE DEPARTMENTS (
                            DEPT_ID_DEP CHAR(9) NOT NULL,
                            DEP_NAME VARCHAR(15) ,
                            MANAGER_ID CHAR(9),
                            LOC_ID CHAR(9),
                            PRIMARY KEY (DEPT_ID_DEP)
                          );

CREATE TABLE LOCATIONS (
                          LOCT_ID CHAR(9) NOT NULL,
                          DEP_ID_LOC CHAR(9) NOT NULL,
                          PRIMARY KEY (LOCT_ID,DEP_ID_LOC)
                        );


LOAD DATA LOCAL INFILE '../HR_DB_CSV_Files/Employees_updated.csv' 
INTO TABLE EMPLOYEES
FIELDS TERMINATED BY ',';

LOAD DATA LOCAL INFILE '../HR_DB_CSV_Files/JobsHistory.csv' 
INTO TABLE JOB_HISTORY
FIELDS TERMINATED BY ',';

LOAD DATA LOCAL INFILE '../HR_DB_CSV_Files/Jobs.csv' 
INTO TABLE JOBS
FIELDS TERMINATED BY ',';

LOAD DATA LOCAL INFILE '../HR_DB_CSV_Files/Departments.csv' 
INTO TABLE DEPARTMENTS
FIELDS TERMINATED BY ',';

LOAD DATA LOCAL INFILE '../HR_DB_CSV_Files/Locations.csv' 
INTO TABLE LOCATIONS
FIELDS TERMINATED BY ',';

-- SELECT * FROM EMPLOYEES;D

-- JOIN Practice

-- 1. Select the names and job start date of all employees who work for Dept 5

-- SELECT  E.F_NAME, E.L_NAME, J.START_DATE 
-- FROM EMPLOYEES E JOIN JOB_HISTORY J
-- ON E.EMP_ID = J.EMPL_ID

-- 2. Select names, job start dates, and job titles of all employees who work in dept 5

-- SELECT E.F_NAME, E.L_NAME, H.START_DATE, J.JOB_TITLE
-- FROM EMPLOYEES E 
-- JOIN JOB_HISTORY H ON E.EMP_ID = H.EMPL_ID
-- JOIN JOBS J ON H.JOBS_ID = J.JOB_IDENT

-- 3. Do a LEFT OUTER JOIN on the employees and department tables and select employee id and dept name for all employees

-- SELECT E.EMP_ID, D.DEP_NAME
-- FROM EMPLOYEES E
-- LEFT JOIN DEPARTMENTS D ON E.DEP_ID = D.DEPT_ID_DEP

-- 4. Question# 3, but only display employees born before 1980

-- SELECT E.EMP_ID, D.DEP_NAME, E.B_DATE
-- FROM EMPLOYEES E 
-- LEFT JOIN DEPARTMENTS D ON E.DEP_ID = D.DEPT_ID_DEP
-- WHERE YEAR (E.B_DATE) < 1980
-- ORDER BY E.B_DATE DESC; 

-- 5. Question# 4, but have the result set include all the employees, but only displays department names for employees born before 1980

-- SELECT E.EMP_ID, D.DEP_NAME, E.B_DATE
-- FROM EMPLOYEES E 
-- LEFT JOIN DEPARTMENTS D ON E.DEP_ID = D.DEPT_ID_DEP
-- AND YEAR (E.B_DATE) < 1980
-- ORDER BY E.B_DATE DESC; 

-- 6. Do a full JOIN on the employees and department tables and select the first name, last name and dept name of all employees.

-- SELECT E.F_NAME, E.L_NAME, D.DEP_NAME
-- FROM EMPLOYEES E
-- JOIN DEPARTMENTS D ON E.DEP_ID = D.DEPT_ID_DEP;

-- OR --

-- select E.F_NAME,E.L_NAME,D.DEP_NAME
-- from EMPLOYEES AS E 
-- LEFT OUTER JOIN DEPARTMENTS AS D ON E.DEP_ID=D.DEPT_ID_DEP

-- UNION

-- select E.F_NAME,E.L_NAME,D.DEP_NAME
-- from EMPLOYEES AS E 
-- RIGHT OUTER JOIN DEPARTMENTS AS D ON E.DEP_ID=D.DEPT_ID_DEP

-- 7. Question# 6, but have result set include all employee names but dept_id and dept_name for only male employees.

SELECT E.F_NAME, E.L_NAME, E.SEX, D.DEP_NAME
FROM EMPLOYEES E
JOIN DEPARTMENTS D ON E.DEP_ID = D.DEPT_ID_DEP
AND SEX = 'M';

-- SOURCE JOINS.SQL;