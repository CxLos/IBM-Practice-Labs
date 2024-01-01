USE ibm_pets_db;

DROP TABLE IF EXISTS ShoeShop;

CREATE TABLE ShoeShop (
    Product VARCHAR(25) NOT NULL,
    Stock INTEGER NOT NULL,
    Price DECIMAL(8,2)
    -- CHECK(Price>0) 
    NOT NULL,
    PRIMARY KEY (Product)
    );

INSERT INTO ShoeShop VALUES
('Boots',11,200),
('High heels',8,600),
('Brogues',10,150),
('Trainers',14,300);

-- SELECT * FROM BankAccounts;
-- SELECT * FROM ShoeShop;

-- Rose Transaction

DELIMITER //

CREATE PROCEDURE TRANSACTION_ROSE()
BEGIN
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        RESIGNAL;
    END;
    
    START TRANSACTION;

    UPDATE BankAccounts
    SET Balance = Balance-200
    WHERE AccountName = 'Rose';

    UPDATE BankAccounts
    SET Balance = Balance+200
    WHERE AccountName = 'Shoe Shop';

    UPDATE ShoeShop
    SET Stock = Stock-1
    WHERE Product = 'Boots';

    COMMIT;
END //

DELIMITER ;

-- James Transaction

DELIMITER //

CREATE PROCEDURE TRANSACTION_JAMES()

BEGIN

    DECLARE EXIT HANDLER FOR SQLEXCEPTION

    BEGIN
        ROLLBACK;
        RESIGNAL;
    END;

    START TRANSACTION;

    UPDATE BankAccounts
    SET Balance = Balance-300
    WHERE AccountName = 'James';

    UPDATE BankAccounts
    SET Balance = Balance+300
    WHERE AccountName = 'Shoe Shop';

    UPDATE ShoeShop
    SET Stock = Stock-1
    WHERE Product = 'Trainers';

    COMMIT;
END //

-- DELIMITER ;

-- CALL TRANSACTION_ROSE;
-- DROP PROCEDURE TRANSACTION_ROSE;
CALL TRANSACTION_JAMES;

SELECT * FROM BankAccounts;

SELECT * FROM ShoeShop;

-- source ShoeShop.sql;