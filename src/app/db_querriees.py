from sqlalchemy import create_engine, text
import pandas as pd


class db_connector:
    def __init__(self, DB_URL: str):
        self.engine = create_engine(DB_URL)

    def connection_alive(self):
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                return True
        except Exception as e:
            print(f"Database unavailable: {type(e).__name__}: {e}")
            return False

    def insert_history(self, row: dict) -> int:
        with self.engine.begin() as conn:
            result = conn.execute(
                text("""
                    INSERT INTO history (
                        model_train_run_id,
                        Age,
                        Attrition,
                        BusinessTravel,
                        DailyRate,
                        Department,
                        DistanceFromHome,
                        Education,
                        EducationField,
                        EmployeeCount,
                        EmployeeNumber,
                        EnvironmentSatisfaction,
                        Gender,
                        HourlyRate,
                        JobInvolvement,
                        JobLevel,
                        JobRole,
                        JobSatisfaction,
                        MaritalStatus,
                        MonthlyIncome,
                        MonthlyRate,
                        NumCompaniesWorked,
                        Over18,
                        OverTime,
                        PercentSalaryHike,
                        PerformanceRating,
                        RelationshipSatisfaction,
                        StandardHours,
                        StockOptionLevel,
                        TotalWorkingYears,
                        TrainingTimesLastYear,
                        WorkLifeBalance,
                        YearsAtCompany,
                        YearsInCurrentRole,
                        YearsSinceLastPromotion,
                        YearsWithCurrManager
                    )
                    VALUES (
                        :model_train_run_id,
                        :Age,
                        :Attrition,
                        :BusinessTravel,
                        :DailyRate,
                        :Department,
                        :DistanceFromHome,
                        :Education,
                        :EducationField,
                        :EmployeeCount,
                        :EmployeeNumber,
                        :EnvironmentSatisfaction,
                        :Gender,
                        :HourlyRate,
                        :JobInvolvement,
                        :JobLevel,
                        :JobRole,
                        :JobSatisfaction,
                        :MaritalStatus,
                        :MonthlyIncome,
                        :MonthlyRate,
                        :NumCompaniesWorked,
                        :Over18,
                        :OverTime,
                        :PercentSalaryHike,
                        :PerformanceRating,
                        :RelationshipSatisfaction,
                        :StandardHours,
                        :StockOptionLevel,
                        :TotalWorkingYears,
                        :TrainingTimesLastYear,
                        :WorkLifeBalance,
                        :YearsAtCompany,
                        :YearsInCurrentRole,
                        :YearsSinceLastPromotion,
                        :YearsWithCurrManager
                    )
                    RETURNING history.row_id
                """),
                row,
            )
            row_id = result.scalar_one()
            return row_id

    def insert_train_history(self, row: dict):
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO train_history (
                        model_train_run_id,
                        row_id,
                        pipeline_shema_version,
                        data_part
                    )
                    VALUES (
                        :model_train_run_id,
                        :row_id,
                        :pipeline_shema_version,
                        :data_part
                    )
                """),
                row,
            )

    def insert_gt_labels(self, row: dict):
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO gt_labels (
                        row_id,
                        GT_Attrition
                    )
                    VALUES (
                        :row_id,
                        :GT_Attrition
                    )
                """),
                row,
            )

    def count_unseen_requests(self):
        with self.engine.begin() as conn:
            result = conn.execute(text("""
                SELECT
                    (
                    SELECT
                        COUNT(*)
                    FROM
                        history) - (
                    SELECT
                        MAX(w_stop)
                    FROM
                        drift);
            """))
        return result.scalar()

    def get_unseen_requests(self):
        with self.engine.begin() as conn:
            result = conn.execute(text("""
                SELECT 
                    row_id,
                    model_train_run_id,
                    age AS "Age",
                    attrition AS "Attrition",
                    businesstravel AS "BusinessTravel",
                    dailyrate AS "DailyRate",
                    department AS "Department",
                    distancefromhome AS "DistanceFromHome",
                    education AS "Education",
                    educationfield AS "EducationField",
                    employeecount AS "EmployeeCount",
                    employeenumber AS "EmployeeNumber",
                    environmentsatisfaction AS "EnvironmentSatisfaction",
                    gender AS "Gender",
                    hourlyrate AS "HourlyRate",
                    jobinvolvement AS "JobInvolvement",
                    joblevel AS "JobLevel",
                    jobrole AS "JobRole",
                    jobsatisfaction AS "JobSatisfaction",
                    maritalstatus AS "MaritalStatus",
                    monthlyincome AS "MonthlyIncome",
                    monthlyrate AS "MonthlyRate",
                    numcompaniesworked AS "NumCompaniesWorked",
                    over18 AS "Over18",
                    overtime AS "OverTime",
                    percentsalaryhike AS "PercentSalaryHike",
                    performancerating AS "PerformanceRating",
                    relationshipsatisfaction AS "RelationshipSatisfaction",
                    standardhours AS "StandardHours",
                    stockoptionlevel AS "StockOptionLevel",
                    totalworkingyears AS "TotalWorkingYears",
                    trainingtimeslastyear AS "TrainingTimesLastYear",
                    worklifebalance AS "WorkLifeBalance",
                    yearsatcompany AS "YearsAtCompany",
                    yearsincurrentrole AS "YearsInCurrentRole",
                    yearssincelastpromotion AS "YearsSinceLastPromotion",
                    yearswithcurrmanager AS "YearsWithCurrManager"
                FROM
                    history
                WHERE
                    row_id > (
                    SELECT
                        MAX(w_stop)
                    FROM
                        drift)
                ORDER BY row_id
                LIMIT(1470)
            """))
        return pd.DataFrame(result.fetchall(), columns=result.keys())

    def get_earliest_window_with_new_labels(self):
        with self.engine.begin() as conn:
            result = conn.execute(text("""
                SELECT
                    *
                FROM
                    drift
                WHERE
                    drift.window_id = (
                    SELECT
                        min(window_id)
                    FROM
                        drift
                    WHERE
                        (target_drift IS NULL
                        OR concept_drift IS NULL)
                        AND
                    1470 = (
                        SELECT
                            count(*)
                        FROM
                            gt_labels
                        WHERE
                            row_id >= drift.w_start
                            AND row_id <= drift.w_stop))
            """))
        return pd.DataFrame(result.fetchall(), columns=result.keys())

    def get_window_rows(self, w_start: int, w_stop: int):
        with self.engine.begin() as conn:
            result = conn.execute(
                text("""
                SELECT
                    history.row_id AS row_id,
                    age AS "Age",
                    attrition AS "Attrition",
                    businesstravel AS "BusinessTravel",
                    dailyrate AS "DailyRate",
                    department AS "Department",
                    distancefromhome AS "DistanceFromHome",
                    education AS "Education",
                    educationfield AS "EducationField",
                    employeecount AS "EmployeeCount",
                    employeenumber AS "EmployeeNumber",
                    environmentsatisfaction AS "EnvironmentSatisfaction",
                    gender AS "Gender",
                    hourlyrate AS "HourlyRate",
                    jobinvolvement AS "JobInvolvement",
                    joblevel AS "JobLevel",
                    jobrole AS "JobRole",
                    jobsatisfaction AS "JobSatisfaction",
                    maritalstatus AS "MaritalStatus",
                    monthlyincome AS "MonthlyIncome",
                    monthlyrate AS "MonthlyRate",
                    numcompaniesworked AS "NumCompaniesWorked",
                    over18 AS "Over18",
                    overtime AS "OverTime",
                    percentsalaryhike AS "PercentSalaryHike",
                    performancerating AS "PerformanceRating",
                    relationshipsatisfaction AS "RelationshipSatisfaction",
                    standardhours AS "StandardHours",
                    stockoptionlevel AS "StockOptionLevel",
                    totalworkingyears AS "TotalWorkingYears",
                    trainingtimeslastyear AS "TrainingTimesLastYear",
                    worklifebalance AS "WorkLifeBalance",
                    yearsatcompany AS "YearsAtCompany",
                    yearsincurrentrole AS "YearsInCurrentRole",
                    yearssincelastpromotion AS "YearsSinceLastPromotion",
                    yearswithcurrmanager AS "YearsWithCurrManager",
                    gt_attrition AS "GT_Attrition"
                FROM
                    history
                JOIN gt_labels ON
                    history.row_id = gt_labels.row_id
                WHERE
                    history.row_id >= :w_start
                    AND history.row_id <= :w_stop;
            """),
                {"w_start": w_start, "w_stop": w_stop},
            )
        return pd.DataFrame(result.fetchall(), columns=result.keys())

    def get_train_data(self, run_id: str):
        with self.engine.begin() as conn:
            result = conn.execute(
                text("""
                SELECT
                    gt_labels.row_id AS row_id,
                    model_train_run_id,
                    pipeline_shema_version,                   
                    age AS "Age",
                    attrition AS "Attrition",
                    businesstravel AS "BusinessTravel",
                    dailyrate AS "DailyRate",
                    department AS "Department",
                    distancefromhome AS "DistanceFromHome",
                    education AS "Education",
                    educationfield AS "EducationField",
                    employeecount AS "EmployeeCount",
                    employeenumber AS "EmployeeNumber",
                    environmentsatisfaction AS "EnvironmentSatisfaction",
                    gender AS "Gender",
                    hourlyrate AS "HourlyRate",
                    jobinvolvement AS "JobInvolvement",
                    joblevel AS "JobLevel",
                    jobrole AS "JobRole",
                    jobsatisfaction AS "JobSatisfaction",
                    maritalstatus AS "MaritalStatus",
                    monthlyincome AS "MonthlyIncome",
                    monthlyrate AS "MonthlyRate",
                    numcompaniesworked AS "NumCompaniesWorked",
                    over18 AS "Over18",
                    overtime AS "OverTime",
                    percentsalaryhike AS "PercentSalaryHike",
                    performancerating AS "PerformanceRating",
                    relationshipsatisfaction AS "RelationshipSatisfaction",
                    standardhours AS "StandardHours",
                    stockoptionlevel AS "StockOptionLevel",
                    totalworkingyears AS "TotalWorkingYears",
                    trainingtimeslastyear AS "TrainingTimesLastYear",
                    worklifebalance AS "WorkLifeBalance",
                    yearsatcompany AS "YearsAtCompany",
                    yearsincurrentrole AS "YearsInCurrentRole",
                    yearssincelastpromotion AS "YearsSinceLastPromotion",
                    yearswithcurrmanager AS "YearsWithCurrManager",
                    gt_attrition AS "GT_Attrition"
                FROM
                    (history
                JOIN (
                    SELECT
                        row_id,
                        pipeline_shema_version
                    FROM
                        train_history
                    WHERE
                        model_train_run_id = :run_id) AS train_data ON
                    history.row_id = train_data.row_id)
                JOIN gt_labels ON
                    history.row_id = gt_labels.row_id
                LIMIT(1470);

            """),
                {"run_id": run_id},
            )
        return pd.DataFrame(result.fetchall(), columns=result.keys())

    def insert_drift_report(self, data: dict):
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                INSERT
                    INTO
                    drift (w_start,
                    w_stop,
                    run_id,
                    data_drift,
                    target_drift,
                    concept_drift,
                    trained)
                VALUES (:w_start,
                :w_stop,
                :run_id,
                :data_drift,
                :target_drift,
                :concept_drift,
                :trained);
                    """),
                data,
            )

    def update_drift_report(
        self,
        window_id: int,
        target_drift: bool = None,
        concept_drift: bool = None,
        trained: bool = None,
    ):
        with self.engine.begin() as conn:
            result = conn.execute(
                text("""
                UPDATE drift
                SET
                    target_drift = COALESCE(:target_drift, target_drift),
                    concept_drift = COALESCE(:concept_drift, concept_drift),
                    trained = COALESCE(:trained, trained)
                WHERE window_id = :id;
                    """),
                {
                    "id": window_id,
                    "target_drift": target_drift,
                    "concept_drift": concept_drift,
                    "trained": trained,
                },
            )

    def get_window_for_retrain(self):
        with self.engine.begin() as conn:
            result = conn.execute(text("""
                SELECT
                    *
                FROM
                    drift
                WHERE
                    ((target_drift = TRUE)
                        OR (concept_drift = TRUE)
                            OR (data_drift = TRUE))
                    AND (trained = FALSE)
                    AND (concept_drift IS NOT NULL
                        OR target_drift IS NOT NULL)
                ORDER BY
                    window_id DESC
                LIMIT(1);
                    """))
        return pd.DataFrame(result.fetchall(), columns=result.keys())

    def blu(self):
        with self.engine.begin() as conn:
            result = conn.execute(text("""

                    """))
        return pd.DataFrame(result.fetchall(), columns=result.keys())
