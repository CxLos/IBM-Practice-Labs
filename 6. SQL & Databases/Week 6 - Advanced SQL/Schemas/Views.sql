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

SELECT * FROM EMPLOYEES;
-- SELECT * FROM JOBS;
-- SELECT * FROM JOB_HISTORY;
SELECT * FROM DEPARTMENTS;
-- SELECT * FROM LOCATIONS;

-- SOURCE HR_DB2.SQL;

-- PRACTICE

-- 1. CREATE VIEW

-- CREATE VIEW EMPSALARY AS
-- SELECT EMP_ID, F_NAME, L_NAME, B_DATE, SEX, SALARY
-- FROM EMPLOYEES;
-- SELECT * FROM EMPSALARY;

-- 2. UPDATE VIEW

-- CREATE OR REPLACE VIEW EMPSALARY AS
-- SELECT EMP_ID, F_NAME, L_NAME, B_DATE, SEX, JOB_TITLE,
-- MIN_SALARY, MAX_SALARY
-- FROM EMPLOYEES, JOBS
-- WHERE EMPLOYEES.JOB_ID = JOBS.JOB_IDENT;
-- SELECT * FROM EMPSALARY;

-- 3. DROP VIEW

-- DROP VIEW EMPSALARY;

-- 1. Create a view “EMP_DEPT” which has the following information.

CREATE VIEW EMP_DEPT AS
SELECT EMP_ID, F_NAME, L_NAME, DEP_ID
FROM EMPLOYEES;
-- SELECT * FROM EMP_DEPT;

-- 2. Modify “EMP_DEPT” such that it displays Department names instead of Department IDs. For this, we need to combine information from EMPLOYEES and DEPARTMENTS as follows.

CREATE OR REPLACE VIEW EMP_DEPT AS
SELECT EMP_ID, F_NAME, L_NAME, DEP_NAME
FROM EMPLOYEES, DEPARTMENTS
WHERE EMPLOYEES.DEP_ID = DEPARTMENTS.DEPT_ID_DEP;
SELECT * FROM EMP_DEPT;

-- 3. Drop the view “EPM_DEPT”.

-- DROP VIEW EMP_DEPT